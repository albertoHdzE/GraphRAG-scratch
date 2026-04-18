import numpy as np
import time
import tiktoken
from typing import List, Dict, Optional
from langfuse.openai import OpenAI
from sklearn.metrics import silhouette_score

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class MetrologyEngine:
    def __init__(self, model_name: str = "google/gemma-3-4b"):
        self.model_name = model_name
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Accurately count tokens using tiktoken.
        """
        return len(self.encoder.encode(text))

    def calculate_information_density(self, chunks: List[str], summary: str, trace_id: Optional[str] = None) -> float:
        """
        Measures semantic preservation: (Preserved Concepts) / (Source Token Count).
        Normalized to reflect density per 100 tokens.
        """
        chunks_text = "\n".join(chunks[:5])
        extraction_prompt = f"""
        Extract the 5 most critical legal entities or clauses from this text.
        Text: {chunks_text}
        Format: entity1, entity2, ...
        No icons.
        """
        
        try:
            res_chunks = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                trace_id=trace_id,
                name="Metrology: Entity Extraction"
            )
            raw_entities = res_chunks.choices[0].message.content
            
            # Robust parsing: Remove bullets, numbers, and split by common delimiters
            import re
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", raw_entities, flags=re.MULTILINE) # Remove 1. or 1)
            cleaned = re.sub(r"^\*\s*", "", cleaned, flags=re.MULTILINE) # Remove bullets
            source_entities = [e.strip().lower() for e in re.split(r"[\n,;]", cleaned) if e.strip()]
            
            summary_lower = summary.lower()
            preserved_count = sum(1 for en in source_entities if en in summary_lower)
            
            summary_tokens = self.count_tokens(summary)
            if summary_tokens == 0: return 0.0
            
            # Density Score: Preserved Entities per 100 tokens
            return (preserved_count / summary_tokens) * 100
        except Exception:
            return 0.0

    def calculate_faithfulness_score(self, chunks: List[str], summary: str, trace_id: Optional[str] = None) -> float:
        """
        LLM-as-a-Judge: Gemma verifies summary claims against source text.
        """
        source_context = "\n\n".join(chunks[:3])
        eval_prompt = f"""
        Source Text:
        {source_context}

        Summary to Evaluate:
        {summary}

        Task: Rate the 'Faithfulness' of the summary on a scale of 1 to 10. 
        A score of 10 means every claim in the summary is perfectly grounded in the source text.
        A score of 1 means the summary contains hallucinations or contradicts the source.
        Only output the numerical score.
        No icons.
        """
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                trace_id=trace_id,
                name="Metrology: Faithfulness Audit"
            )
            score_text = response.choices[0].message.content.strip()
            # Extract first number found
            import re
            match = re.search(r"(\d+)", score_text)
            return float(match.group(1)) if match else 5.0
        except Exception:
            return 5.0

    def calculate_clustering_metrics(self, embeddings: np.ndarray, labels: List[int]) -> Dict:
        """
        Calculates mathematical clustering quality.
        """
        try:
            if len(set(labels)) < 2:
                return {"silhouette": 0.0, "purity": 1.0}
            
            s_score = silhouette_score(embeddings, labels)
            
            # Purity logic (Intra-cluster similarity ratio)
            return {
                "silhouette": float(s_score),
                "purity": float((s_score + 1) * 5) # Normalize -1..1 to 0..10
            }
        except Exception:
            return {"silhouette": 0.0, "purity": 0.0}

    def generate_rich_metrics(self, results: Dict) -> Dict:
        """
        Consolidated metrology report with token counts and LLM evaluations.
        """
        chunks = results["chunks"]
        
        # Path A Metrics
        rag_summary = results["rag"]["summary"]
        rag_trace_id = results["rag"]["trace_id"]
        rag_tokens = self.count_tokens(rag_summary)
        rag_faithfulness = self.calculate_faithfulness_score(chunks, rag_summary, trace_id=rag_trace_id)
        rag_density = self.calculate_information_density(chunks, rag_summary, trace_id=rag_trace_id)
        
        # Path B Metrics
        rau_summary = results["rau"]["summary"]
        rau_trace_id = results["rau"]["trace_id"]
        rau_tokens = self.count_tokens(rau_summary)
        rau_faithfulness = self.calculate_faithfulness_score(chunks, rau_summary, trace_id=rau_trace_id)
        rau_density = self.calculate_information_density(chunks, rau_summary, trace_id=rau_trace_id)
        
        # RAU Matrix Metrics
        rau_metrics = self.calculate_clustering_metrics(
            np.array(results["rau"]["embeddings"]), 
            results["rau"]["labels"]
        )
        
        return {
            "naive_rag": {
                "latency_ms": results["rag"]["latency"],
                "total_tokens": rag_tokens,
                "info_density": rag_density,
                "faithfulness": rag_faithfulness,
                "efficiency": rag_tokens / (results["rag"]["latency"] / 1000)
            },
            "structured_rau": {
                "latency_ms": results["rau"]["latency"],
                "total_tokens": rau_tokens,
                "info_density": rau_density,
                "faithfulness": rau_faithfulness,
                "silhouette": rau_metrics["silhouette"],
                "purity": rau_metrics["purity"],
                "efficiency": rau_tokens / (results["rau"]["latency"] / 1000)
            }
        }

class ExpertInterpreter:
    def __init__(self, model_name: str = "google/gemma-3-4b"):
        self.model_name = model_name

    def interpret_results(self, metrics: Dict, results: Dict, trace_id: Optional[str] = None) -> str:
        """
        Uses Gemma to analyze the numerical data and Path A/B summaries.
        Returns a structured legal-AI expert analysis.
        """
        rag_metrics = metrics["naive_rag"]
        rau_metrics = metrics["structured_rau"]
        
        prompt = f"""
        As a Legal AI Expert, analyze this RAG system comparison:
        
        System A (Naive RAG):
        - Info Density: {rag_metrics['info_density']:.2f}
        - Faithfulness: {rag_metrics['faithfulness']:.1f}/10
        - Efficiency: {rag_metrics['efficiency']:.1f} t/s
        
        System B (Structured RAU):
        - Info Density: {rau_metrics['info_density']:.2f}
        - Faithfulness: {rau_metrics['faithfulness']:.1f}/10
        - Efficiency: {rau_metrics['efficiency']:.1f} t/s
        - Clustering Quality (Silhouette): {rau_metrics['silhouette']:.3f}
        
        Summaries Found:
        Path A: {results['rag']['summary'][:500]}...
        Path B: {results['rau']['summary'][:500]}...
        
        Task: Provide a deep interpretation of these results. Explain:
        1. Which path preserves more legal information?
        2. Is the RAU clustering mathematically sound based on the silhouette score?
        3. A final recommendation on which synthesis is more reliable for identifying subtle contract risks.
        
        Style: Professional, analytical, alphanumeric only. No icons or emojis.
        """
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                trace_id=trace_id,
                name="Mission Expert Interpretation"
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating interpretation: {str(e)}"
