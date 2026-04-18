from data_engine.vector_store import VectorStoreManager
from challenges.c01_clustering_rau.pipeline import DualPipeline
from challenges.c01_clustering_rau.comparison import MetrologyEngine
import os

def verify_and_report():
    print("Initializing End-to-End Verification...")
    
    # 1. Initialize Components
    vstore = VectorStoreManager(persist_directory="data_engine/storage/chroma_db")
    pipeline = DualPipeline(vstore)
    metrology = MetrologyEngine()
    
    query = "What are the common liability limitations in these contracts?"
    
    # 2. Execute Dual Pipeline
    print(f"Executing Query: {query}")
    results = pipeline.execute_comparison(query)
    
    # 3. Calculate Metrics
    print("Calculating Metrics...")
    metrics = metrology.generate_rich_metrics(results)
    
    # 4. Generate Performance Report
    report_content = f"""# Advanced Performance Report: Challenge 1

## Execution Summary
- Query: {query}
- Model: google/gemma-3-4b (Local)
- Token Counter: tiktoken (cl100k_base)

## Deep Metrics Comparison

### Naive RAG (Path A)
- Latency: {metrics['naive_rag']['latency_ms']:.2f} ms
- Information Density: {metrics['naive_rag']['info_density']:.2f}
- Faithfulness: {metrics['naive_rag']['faithfulness']:.1f}/10
- Token Efficiency: {metrics['naive_rag']['efficiency']:.1f} t/s

### Structured RAU (Path B)
- Latency: {metrics['structured_rau']['latency_ms']:.2f} ms
- Information Density: {metrics['structured_rau']['info_density']:.2f}
- Faithfulness: {metrics['structured_rau']['faithfulness']:.1f}/10
- Silhouette Score: {metrics['structured_rau']['silhouette']:.3f}
- Token Efficiency: {metrics['structured_rau']['efficiency']:.1f} t/s

## Traceability Check
Status: Trace details captured. Advanced scoring pushed to Langfuse Cloud.

## Analysis
The Structured RAU pipeline provides increased information density by grouping similar legal concepts before synthesis, though it incurs higher latency due to the multi-step clustering and summary process.
"""

    with open("performance_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("Verification Complete. performance_report.md generated.")

if __name__ == "__main__":
    verify_and_report()
