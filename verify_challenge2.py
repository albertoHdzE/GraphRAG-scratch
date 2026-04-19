import os

from dotenv import load_dotenv

from data_engine.vector_store import VectorStoreManager
from challenges.c02_rag_graphrag.pipeline import DualPipeline
from challenges.c02_rag_graphrag.comparison import MetrologyEngine


def verify():
    load_dotenv()

    vstore = VectorStoreManager(persist_directory="data_engine/storage/chroma_db")
    pipeline = DualPipeline(vstore)
    metrology = MetrologyEngine()

    query = "Identify risks involving both termination and liability clauses"

    results = pipeline.execute_comparison(query)
    assert "rag" in results and "rau" in results and "graphrag" in results

    metrics = metrology.generate_rich_metrics(results)
    assert "naive_rag" in metrics and "structured_rau" in metrics
    assert "graphrag" in metrics

    g = results["graphrag"]
    gmeta = g.get("metadata", {}) or {}
    assert isinstance(gmeta.get("graph_nodes_count"), int)
    assert isinstance(gmeta.get("graph_edges_count"), int)
    assert isinstance(gmeta.get("traversal_depth"), int)

    print("Challenge 2 verification passed")
    print(
        {
            "rag_latency_ms": metrics["naive_rag"]["latency_ms"],
            "rau_latency_ms": metrics["structured_rau"]["latency_ms"],
            "graphrag_latency_ms": metrics["graphrag"]["latency_ms"],
            "graphrag_graph_nodes_count": gmeta.get("graph_nodes_count"),
            "graphrag_graph_edges_count": gmeta.get("graph_edges_count"),
            "graphrag_traversal_depth": gmeta.get("traversal_depth"),
            "graphrag_graph_coverage": gmeta.get("graph_coverage"),
            "graphrag_path_coherence": gmeta.get("path_coherence"),
            "graphrag_llm_ok": gmeta.get("llm_ok"),
        }
    )


if __name__ == "__main__":
    verify()

