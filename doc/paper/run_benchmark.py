import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from data_engine.vector_store import VectorStoreManager
from challenges.c02_rag_graphrag.pipeline import DualPipeline
from challenges.c02_rag_graphrag.comparison import MetrologyEngine


QUERIES = [
    "Identify risks involving both termination and liability clauses",
    "What obligations survive termination and how do they interact with limitation of liability",
    "Find conflicts between indemnification and limitation of liability",
    "Identify risks where confidentiality obligations collide with disclosure or audit rights",
    "Assess whether termination triggers payment obligations or refund obligations",
    "Summarize termination for convenience and termination for cause",
    "Summarize limitation of liability and excluded damages",
    "Summarize indemnity and defense obligations",
    "Summarize confidentiality scope, exclusions, and duration",
    "Summarize governing law and venue",
    "Summarize assignment restrictions and change of control",
    "Summarize force majeure conditions and notice requirements",
    "Summarize warranty disclaimers and remedy limitations",
    "Summarize payment terms and late payment penalties",
    "Summarize dispute resolution, arbitration, and escalation steps",
    "Summarize IP ownership and license grants",
    "Summarize audit rights and record retention requirements",
    "Summarize non-solicitation and non-compete restrictions",
    "Summarize data security obligations and breach notification requirements",
    "Summarize insurance requirements and proof of coverage",
    "Identify clauses that could create unlimited liability exposure",
    "Identify clauses that could allow unilateral changes to pricing or scope",
    "Identify clauses that shift regulatory compliance risk to one party",
    "Identify clauses that create unclear acceptance criteria or deliverables",
    "Identify clauses that create broad termination rights with short notice",
]

FULL_METRICS_QUERIES = [
    "Identify risks involving both termination and liability clauses",
    "What obligations survive termination and how do they interact with limitation of liability",
    "Find conflicts between indemnification and limitation of liability",
    "Identify risks where confidentiality obligations collide with disclosure or audit rights",
    "Assess whether termination triggers payment obligations or refund obligations",
    "Identify clauses that could create unlimited liability exposure",
    "Identify clauses that create broad termination rights with short notice",
]


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(output_dir: Path) -> None:
    load_dotenv()

    _mkdir(output_dir)

    vstore = VectorStoreManager(persist_directory="data_engine/storage/chroma_db")
    pipeline = DualPipeline(vstore)
    metrology = MetrologyEngine()
    pipeline._ensure_graph_ready()

    retrieval_csv = output_dir / "results_metrics.csv"
    systems_csv = output_dir / "results_systems.csv"

    with retrieval_csv.open("w", newline="", encoding="utf-8") as f_ret, systems_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_sys:
        ret_writer = csv.DictWriter(
            f_ret,
            fieldnames=[
                "query",
                "seed_latency_ms",
                "graph_latency_ms",
                "retrieval_latency_ms",
                "traversal_depth",
                "graph_nodes",
                "graph_edges",
                "graph_coverage",
                "path_coherence",
            ],
        )
        ret_writer.writeheader()

        sys_writer = csv.DictWriter(
            f_sys,
            fieldnames=[
                "query",
                "system",
                "latency_ms",
                "total_tokens",
                "info_density",
                "faithfulness",
                "efficiency",
                "graph_coverage",
                "path_coherence",
                "traversal_depth",
                "graph_nodes",
                "graph_edges",
                "llm_ok",
                "trace_url",
            ],
        )
        sys_writer.writeheader()

        for q in QUERIES:
            retrieval = pipeline._graph_retriever.retrieve(
                q,
                top_k=10,
                depth=2,
                alpha=0.75,
                beta=0.25,
                max_nodes=40,
                edge_prune_threshold=0.60,
            )
            gmd = retrieval.get("metadata", {}) or {}
            ret_writer.writerow(
                {
                    "query": q,
                    "seed_latency_ms": float(gmd.get("seed_latency_ms", 0.0)),
                    "graph_latency_ms": float(gmd.get("graph_latency_ms", 0.0)),
                    "retrieval_latency_ms": float(gmd.get("retrieval_latency_ms", 0.0)),
                    "traversal_depth": int(gmd.get("traversal_depth", 0)),
                    "graph_nodes": int(gmd.get("graph_nodes_count", 0)),
                    "graph_edges": int(gmd.get("graph_edges_count", 0)),
                    "graph_coverage": float(gmd.get("graph_coverage", 0.0)),
                    "path_coherence": float(gmd.get("path_coherence", 0.0)),
                }
            )

            if q not in FULL_METRICS_QUERIES:
                continue

            t0 = time.time()
            results = pipeline.execute_comparison(q)
            metrics = metrology.generate_rich_metrics(results)
            _ = time.time() - t0

            sys_writer.writerow(
                {
                    "query": q,
                    "system": "rag",
                    "latency_ms": float(metrics["naive_rag"]["latency_ms"]),
                    "total_tokens": int(metrics["naive_rag"]["total_tokens"]),
                    "info_density": float(metrics["naive_rag"]["info_density"]),
                    "faithfulness": float(metrics["naive_rag"]["faithfulness"]),
                    "efficiency": float(metrics["naive_rag"]["efficiency"]),
                    "graph_coverage": "",
                    "path_coherence": "",
                    "traversal_depth": "",
                    "graph_nodes": "",
                    "graph_edges": "",
                    "llm_ok": "",
                    "trace_url": (results.get("rag", {}) or {}).get("trace_url", ""),
                }
            )

            sys_writer.writerow(
                {
                    "query": q,
                    "system": "rau",
                    "latency_ms": float(metrics["structured_rau"]["latency_ms"]),
                    "total_tokens": int(metrics["structured_rau"]["total_tokens"]),
                    "info_density": float(metrics["structured_rau"]["info_density"]),
                    "faithfulness": float(metrics["structured_rau"]["faithfulness"]),
                    "efficiency": float(metrics["structured_rau"]["efficiency"]),
                    "graph_coverage": "",
                    "path_coherence": "",
                    "traversal_depth": "",
                    "graph_nodes": "",
                    "graph_edges": "",
                    "llm_ok": "",
                    "trace_url": (results.get("rau", {}) or {}).get("trace_url", ""),
                }
            )

            g = metrics.get("graphrag", {}) or {}
            sys_writer.writerow(
                {
                    "query": q,
                    "system": "graphrag",
                    "latency_ms": float(g.get("latency_ms", 0.0)),
                    "total_tokens": int(g.get("total_tokens", 0)),
                    "info_density": float(g.get("info_density", 0.0)),
                    "faithfulness": float(g.get("faithfulness", 0.0)),
                    "efficiency": float(g.get("efficiency", 0.0)),
                    "graph_coverage": float(g.get("graph_coverage", 0.0)),
                    "path_coherence": float(g.get("path_coherence", 0.0)),
                    "traversal_depth": int(g.get("traversal_depth", 0)),
                    "graph_nodes": int(g.get("graph_nodes_count", 0)),
                    "graph_edges": int(g.get("graph_edges_count", 0)),
                    "llm_ok": bool(((results.get("graphrag", {}) or {}).get("metadata", {}) or {}).get("llm_ok", False)),
                    "trace_url": (results.get("graphrag", {}) or {}).get("trace_url", ""),
                }
            )

    print(f"Wrote {retrieval_csv}")
    print(f"Wrote {systems_csv}")


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
