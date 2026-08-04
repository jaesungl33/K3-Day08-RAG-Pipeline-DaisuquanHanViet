"""Streamlit UI for the University Services RAG chatbot.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Trợ lý dịch vụ sinh viên",
    page_icon=":material/school:",
    layout="centered",
    initial_sidebar_state="expanded",
)


SUGGESTIONS = {
    ":material/payments: Tuition fees": "How are tuition fees calculated at RMIT Vietnam?",
    ":material/meeting_room: Book a study room": "How can I book a group study room at the library?",
    ":material/workspace_premium: Scholarships": "What are the eligibility requirements for scholarships at RMIT Vietnam?",
    ":material/home: Accommodation support": "What accommodation support services does RMIT offer students?",
}


def initialize_state() -> None:
    """Initialize per-user state in one place."""
    st.session_state.setdefault("messages", [])


def clear_conversation() -> None:
    st.session_state.messages = []


def build_contextual_query(query: str, use_memory: bool) -> str:
    """Attach a short conversation window so follow-up questions have context."""
    if not use_memory or not st.session_state.messages:
        return query

    recent_messages = st.session_state.messages[-4:]
    history = "\n".join(
        f"{'Người dùng' if message['role'] == 'user' else 'Trợ lý'}: "
        f"{message['content']}"
        for message in recent_messages
    )
    return (
        "Dựa trên lịch sử hội thoại dưới đây, hãy hiểu câu hỏi mới trong đúng ngữ cảnh.\n"
        f"{history}\nCâu hỏi mới: {query}"
    )


def source_title(source: dict[str, Any], index: int) -> str:
    metadata = source.get("metadata") or {}
    raw_name = metadata.get("title") or metadata.get("source") or metadata.get("source_path")
    if not raw_name:
        return f"Nguồn {index}"
    return Path(str(raw_name)).stem.replace("-", " ").replace("_", " ").strip().title()


def render_sources(sources: list[dict[str, Any]], *, message_key: str) -> None:
    """Render retrieved chunks without letting source details dominate the chat."""
    if not sources:
        return

    with st.expander(
        f"Nguồn tham khảo ({len(sources)})",
        icon=":material/library_books:",
    ):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            score = source.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            retrieval_mode = metadata.get("retrieval_mode") or source.get("source") or "RAG"

            with st.container(border=True, key=f"{message_key}_source_{index}"):
                st.markdown(f"**[{index}] {source_title(source, index)}**")
                st.caption(f"Phương thức: {retrieval_mode} · Điểm: {score_text}")

                source_url = metadata.get("source_url")
                if source_url:
                    st.markdown(f"[Mở tài liệu gốc]({source_url})")

                content = str(source.get("content") or "Không có nội dung trích dẫn.").strip()
                preview = content[:700] + ("…" if len(content) > 700 else "")
                st.text(preview)


def render_message(message: dict[str, Any], index: int) -> None:
    role = message.get("role", "assistant")
    avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))
        if role == "assistant":
            render_sources(message.get("sources") or [], message_key=f"history_{index}")


initialize_state()

with st.sidebar:
    st.title("RMIT Student Hub")
    st.caption("Trợ lý tra cứu chính sách và dịch vụ sinh viên từ kho tài liệu nội bộ.")

    st.subheader("Thiết lập truy xuất", anchor=False)
    retrieval_mode_label = st.segmented_control(
        "Phương thức",
        options=["Hybrid", "Dense"],
        default="Hybrid",
        help="Hybrid kết hợp semantic search và BM25; Dense chỉ dùng semantic search.",
    )
    top_k = st.slider(
        "Số nguồn sử dụng",
        min_value=3,
        max_value=10,
        value=5,
        help="Số đoạn tài liệu được đưa vào bước sinh câu trả lời.",
    )
    use_reranking = st.toggle(
        "Bật reranking",
        value=True,
        disabled=retrieval_mode_label == "Dense",
    )
    use_memory = st.toggle(
        "Ghi nhớ hội thoại",
        value=True,
        help="Dùng tối đa 4 tin nhắn gần nhất để hiểu câu hỏi nối tiếp.",
    )

    st.button(
        "Xóa hội thoại",
        icon=":material/delete_sweep:",
        on_click=clear_conversation,
        disabled=not st.session_state.messages,
        width="stretch",
    )

    st.caption("Hybrid retrieval · RRF reranking · Trả lời kèm trích dẫn")


st.title("Trợ lý dịch vụ sinh viên", anchor=False)
st.caption("Hỏi về học phí, học bổng, thư viện, chỗ ở và các dịch vụ tại RMIT Vietnam.")

if not st.session_state.messages:
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.markdown(
            "Xin chào! Mình sẽ tìm trong kho tài liệu và trả lời kèm nguồn tham khảo. "
            "Bạn muốn tìm hiểu điều gì?"
        )

    selected_suggestion = st.pills(
        "Câu hỏi gợi ý",
        options=list(SUGGESTIONS),
        label_visibility="collapsed",
    )
else:
    selected_suggestion = None

for message_index, chat_message in enumerate(st.session_state.messages):
    render_message(chat_message, message_index)

typed_query = st.chat_input(
    "Nhập câu hỏi về chính sách hoặc dịch vụ sinh viên…",
    key="chat_input",
    max_chars=1000,
    submit_mode="disable",
)
query = typed_query or (SUGGESTIONS.get(selected_suggestion) if selected_suggestion else None)

if query:
    query = query.strip()
    contextual_query = build_contextual_query(query, use_memory)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(query)

    retrieval_mode = "hybrid" if retrieval_mode_label == "Hybrid" else "dense_only"

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.status(
                "Đang tìm tài liệu phù hợp…",
                expanded=False,
            ) as status:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(
                    contextual_query,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    use_reranking=use_reranking if retrieval_mode == "hybrid" else False,
                )
                status.update(
                    label="Đã tổng hợp câu trả lời",
                    state="complete",
                    expanded=False,
                )

            answer = response.get("answer") or "Mình chưa tìm thấy câu trả lời phù hợp."
            sources = response.get("sources") or []
            st.markdown(answer)
            render_sources(sources, message_key=f"current_{len(st.session_state.messages)}")
        except Exception as exc:  # Keep the UI responsive when the local pipeline is not ready.
            answer = (
                "Không thể tạo câu trả lời lúc này. Hãy kiểm tra dữ liệu đã được index "
                f"và khóa API trong tệp `.env`.\n\nChi tiết: `{exc}`"
            )
            sources = []
            st.error(answer, icon=":material/error:")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
