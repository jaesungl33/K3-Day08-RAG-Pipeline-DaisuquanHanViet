"""Automated RAGAS evaluation and A/B comparison for the university RAG."""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv


EVALUATION_DIR = Path(__file__).parent
PROJECT_DIR = EVALUATION_DIR.parent.parent
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
CACHE_PATH = EVALUATION_DIR / "evaluation_cache.json"
SCORES_CACHE_PATH = EVALUATION_DIR / "ragas_scores_cache.json"
load_dotenv(PROJECT_DIR / ".env")

GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "text-embedding-3-small"
CONFIGS = {
    "hybrid_rrf": {"retrieval_mode": "hybrid", "use_reranking": True},
    "dense_only": {"retrieval_mode": "dense_only", "use_reranking": False},
}
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def load_golden_dataset() -> list[dict]:
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if len(data) < 15:
        raise ValueError(f"golden_dataset.json requires >=15 cases, found {len(data)}")
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, 1):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Golden case {index} is missing: {sorted(missing)}")
    return data


def _dataset_fingerprint() -> str:
    return hashlib.sha256(GOLDEN_DATASET_PATH.read_bytes()).hexdigest()


def _generation_fingerprint() -> str:
    generation_path = PROJECT_DIR / "src" / "task10_generation.py"
    return hashlib.sha256(generation_path.read_bytes()).hexdigest()


def _load_cache() -> dict:
    fingerprint = _dataset_fingerprint()
    generation_fingerprint = _generation_fingerprint()
    if not CACHE_PATH.exists():
        return {
            "generation_model": GENERATION_MODEL,
            "dataset_fingerprint": fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "configs": {},
        }
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    if (
        cache.get("generation_model") != GENERATION_MODEL
        or cache.get("dataset_fingerprint") != fingerprint
        or cache.get("generation_fingerprint") != generation_fingerprint
    ):
        return {
            "generation_model": GENERATION_MODEL,
            "dataset_fingerprint": fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "configs": {},
        }
    return cache


def collect_pipeline_outputs(golden_dataset: list[dict]) -> dict[str, list[dict]]:
    """Generate answers and checkpoint after every case for safe resume."""
    from src.task10_generation import generate_with_citation

    cache = _load_cache()
    outputs: dict[str, list[dict]] = cache.setdefault("configs", {})
    for config_name, parameters in CONFIGS.items():
        rows = outputs.setdefault(config_name, [])
        for index in range(len(rows), len(golden_dataset)):
            item = golden_dataset[index]
            print(f"[{config_name}] {index + 1}/{len(golden_dataset)}: {item['question']}", flush=True)
            result = generate_with_citation(item["question"], top_k=5, **parameters)
            rows.append(
                {
                    "question": item["question"],
                    "answer": result["answer"],
                    "reference": item["expected_answer"],
                    "expected_context": item["expected_context"],
                    "contexts": [source["content"] for source in result["sources"]],
                    "source_files": [
                        source.get("metadata", {}).get("source", "Unknown")
                        for source in result["sources"]
                    ],
                }
            )
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def evaluate_with_ragas(rows: list[dict]) -> dict:
    """Collect the four required RAGAS metrics for one configuration."""
    # RAGAS 0.4.3 still imports this legacy optional VertexAI module, which was
    # removed by langchain-community 0.4. It is only used for isinstance checks;
    # provide a harmless placeholder because this evaluation uses OpenAI only.
    legacy_vertex_module = "langchain_community.chat_models.vertexai"
    if legacy_vertex_module not in sys.modules:
        try:
            __import__(legacy_vertex_module)
        except ModuleNotFoundError:
            shim = types.ModuleType(legacy_vertex_module)
            shim.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[legacy_vertex_module] = shim

    from openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_llm = llm_factory(JUDGE_MODEL, client=client, temperature=0.0)
    # ResponseRelevancy in RAGAS 0.4 still expects the LangChain
    # embed_query/embed_documents interface.
    judge_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=row["reference"],
            )
            for row in rows
        ]
    )
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=180, max_retries=3, max_wait=30, max_workers=8),
        raise_exceptions=True,
        show_progress=True,
        batch_size=8,
    )
    raw_scores = [dict(score) for score in result.scores]
    normalized_rows = []
    for source_row, score in zip(rows, raw_scores):
        normalized = {
            "question": source_row["question"],
            "faithfulness": _safe_float(score.get("faithfulness")),
            "answer_relevancy": _safe_float(score.get("answer_relevancy")),
            "context_recall": _safe_float(score.get("context_recall")),
            # RAGAS 0.4 exposes the concrete class name instead of the legacy
            # context_precision alias used by earlier releases.
            "context_precision": _safe_float(
                score.get("context_precision", score.get("llm_context_precision_with_reference"))
            ),
        }
        normalized["average"] = _average_metrics(normalized)
        normalized_rows.append(normalized)
    summary = {
        metric: mean(row[metric] for row in normalized_rows)
        for metric in METRIC_NAMES
    }
    summary["average"] = mean(summary.values())
    return {"summary": summary, "rows": normalized_rows}


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    number = float(value)
    return 0.0 if math.isnan(number) else max(0.0, min(1.0, number))


def _average_metrics(row: dict) -> float:
    return mean(row[name] for name in METRIC_NAMES)


def compare_configs(outputs: dict[str, list[dict]]) -> dict:
    cache_key = {
        "generation_model": GENERATION_MODEL,
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "dataset_fingerprint": _dataset_fingerprint(),
        "generation_fingerprint": _generation_fingerprint(),
    }
    if SCORES_CACHE_PATH.exists():
        score_cache = json.loads(SCORES_CACHE_PATH.read_text(encoding="utf-8"))
        if score_cache.get("metadata") != cache_key:
            score_cache = {"metadata": cache_key, "configs": {}}
    else:
        score_cache = {"metadata": cache_key, "configs": {}}

    evaluation = score_cache["configs"]
    for config_name, rows in outputs.items():
        cached = evaluation.get(config_name)
        if cached and len(cached.get("rows", [])) == len(rows):
            print(f"Using cached RAGAS scores for {config_name}", flush=True)
            continue
        print(f"Running RAGAS for {config_name} ({len(rows)} cases)...", flush=True)
        evaluation[config_name] = evaluate_with_ragas(rows)
        SCORES_CACHE_PATH.write_text(
            json.dumps(score_cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return evaluation


def export_results(evaluation: dict, outputs: dict[str, list[dict]]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "# RAGAS Evaluation Results",
        "",
        f"- Generated: `{timestamp}`",
        f"- Golden test cases: **{len(next(iter(outputs.values()))) if outputs else 0}**",
        f"- Generation model: `{GENERATION_MODEL}`",
        f"- RAGAS judge model: `{JUDGE_MODEL}`",
        f"- Embedding model: `{EMBEDDING_MODEL}`",
        "",
        "## Overall A/B comparison",
        "",
        "| Configuration | Faithfulness | Relevancy | Recall | Precision | Average |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in evaluation.items():
        score = result["summary"]
        lines.append(
            f"| {name} | {score['faithfulness']:.3f} | {score['answer_relevancy']:.3f} | "
            f"{score['context_recall']:.3f} | {score['context_precision']:.3f} | {score['average']:.3f} |"
        )

    lines.extend(["", "## Per-question scores", ""])
    for name, result in evaluation.items():
        lines.extend([
            f"### {name}", "",
            "| # | Question | Faithfulness | Relevancy | Recall | Precision | Average |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])
        for index, row in enumerate(result["rows"], 1):
            question = row["question"].replace("|", "\\|")
            lines.append(
                f"| {index} | {question} | {row['faithfulness']:.3f} | "
                f"{row['answer_relevancy']:.3f} | {row['context_recall']:.3f} | "
                f"{row['context_precision']:.3f} | {row['average']:.3f} |"
            )
        lines.append("")

    worst = sorted(
        (
            {"config": config, **row}
            for config, result in evaluation.items()
            for row in result["rows"]
        ),
        key=lambda row: row["average"],
    )[:5]
    lines.extend([
        "## Worst performers", "",
        "| Configuration | Question | Average | Main weakness |",
        "|---|---|---:|---|",
    ])
    for row in worst:
        weakest = min(METRIC_NAMES, key=lambda metric: row[metric])
        lines.append(
            f"| {row['config']} | {row['question'].replace('|', '\\|')} | "
            f"{row['average']:.3f} | {weakest} ({row[weakest]:.3f}) |"
        )

    best_config = max(evaluation, key=lambda name: evaluation[name]["summary"]["average"])
    lines.extend([
        "",
        "## Analysis and recommendations",
        "",
        f"- **{best_config}** achieved the highest overall average in this run.",
        "- Improve low context precision by reducing duplicated PDF chunks and adding Markdown header metadata to the chunker.",
        "- Improve context recall by expanding Vietnamese/English query terms before dense and BM25 retrieval.",
        "- Review the worst questions above and add targeted documents when the corpus lacks explicit evidence.",
        "- Re-run this same golden set after retrieval, chunking, or prompt changes to detect regressions.",
        "",
    ])
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required in .env or the environment")
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} golden test cases", flush=True)
    outputs = collect_pipeline_outputs(golden_dataset)
    evaluation = compare_configs(outputs)
    export_results(evaluation, outputs)
    print(f"Evaluation complete: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
