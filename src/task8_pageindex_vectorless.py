"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    import json
    import time
    from fpdf import FPDF
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache_file = STANDARDIZED_DIR / "doc_ids_cache.json"
    doc_ids = []

    # Đọc cache cũ nếu đã tồn tại
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                doc_ids = json.load(f)
            except json.JSONDecodeError:
                doc_ids = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        pdf_path = md_file.with_suffix(".pdf")
        
        # Convert markdown sang PDF đơn giản bằng fpdf2
        if not pdf_path.exists():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            # Dùng font mặc định (sẽ lỗi font tiếng Việt nếu không nạp font ttf, dùng encode xử lý tạm thời)
            pdf.set_font("Helvetica", size=12)
            
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Ghi nội dung vào PDF, thay thế các ký tự không hỗ trợ
            pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
            pdf.output(str(pdf_path))
            
        print(f"Đang tải lên: {pdf_path.name}...")
        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
                print(f"  ✓ Uploaded: {pdf_path.name} -> {doc_id}")
                
        except Exception as e:
            print(f"  ⚠ Lỗi khi tải lên {pdf_path.name}: {e}")

    # Lưu danh sách doc_ids vào cache JSON
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, indent=4)
    print(f"\nĐã lưu {len(doc_ids)} tài liệu vào {cache_file.name}.")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    import json
    import time
    from pageindex.client import PageIndexClient

    cache_file = STANDARDIZED_DIR / "doc_ids_cache.json"
    
    if not cache_file.exists():
        print("⚠ Cache file chưa tồn tại, vui lòng chạy upload_documents() trước.")
        return []

    with open(cache_file, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    if not doc_ids:
        print("⚠ Không có doc_id nào trong cache.")
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    results = []
    
    # Do PageIndex query thường nhận từng doc_id, ta sẽ dùng doc_id đầu tiên làm ví dụ 
    # (nếu muốn search toàn bộ cần cấu hình theo Collection ID bên phía PageIndex)
    doc_id = doc_ids[0]
    
    try:
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        
        # Poll cho đến khi status == "completed" (Interval 2s, timeout 60s)
        poll_interval = 2
        timeout = 60
        start_time = time.time()
        retrieval = None
        
        while time.time() - start_time < timeout:
            retrieval = client.get_retrieval(retrieval_id)
            status = retrieval.get("status")
            
            if status == "completed":
                break
            elif status in ["failed", "error"]:
                print(f"⚠ Lỗi khi truy vấn PageIndex (Status: {status}).")
                return []
                
            time.sleep(poll_interval)
            
        if not retrieval or retrieval.get("status") != "completed":
            print("⚠ Timeout: Vượt quá thời gian truy vấn PageIndex (60s).")
            return []

        # Parse retrieval["retrieved_nodes"]
        base_score = 1.0
        score_decrement = 0.05

        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    if len(results) >= top_k:
                        return results
                        
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": round(base_score, 3), # PageIndex không có score, gán giảm dần theo rank
                        "metadata": {"section": item.get("section_title", "N/A")},
                        "source": "pageindex",
                    })
                    # Giảm điểm cho kết quả sau
                    base_score = max(0.1, base_score - score_decrement)

    except Exception as e:
        print(f"⚠ Lỗi trong quá trình tìm kiếm PageIndex: {e}")

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
