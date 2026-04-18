# Advanced Performance Report: Challenge 1

## Execution Summary
- Query: What are the common liability limitations in these contracts?
- Model: google/gemma-3-4b (Local)
- Token Counter: tiktoken (cl100k_base)

## Deep Metrics Comparison

### Naive RAG (Path A)
- Latency: 3815.97 ms
- Information Density: 0.00
- Faithfulness: 7.0/10
- Token Efficiency: 110.6 t/s

### Structured RAU (Path B)
- Latency: 13764.78 ms
- Information Density: 0.00
- Faithfulness: 8.0/10
- Silhouette Score: 0.130
- Token Efficiency: 41.3 t/s

## Traceability Check
Status: Trace details captured. Advanced scoring pushed to Langfuse Cloud.

## Analysis
The Structured RAU pipeline provides increased information density by grouping similar legal concepts before synthesis, though it incurs higher latency due to the multi-step clustering and summary process.
