import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_engine.vector_store import VectorStoreManager
from challenges.c01_clustering_rau.pipeline import DualPipeline
from challenges.c01_clustering_rau.comparison import MetrologyEngine, ExpertInterpreter


def render_trace_button(label: str, trace_url: str):
    if isinstance(trace_url, str) and trace_url.startswith("http"):
        st.link_button(label, trace_url)
    else:
        st.caption("Trace link unavailable for this run.")


# Page Config
st.set_page_config(page_title="GraphRAG Mission: Challenge 1 - Advanced", layout="wide")

# Theme / CSS (Premium Soft Blue Aesthetics - Zero Icons Policy)
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* Soft Blue Metrics Cards */
    [data-testid="stMetric"] {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #bbdefb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Dark Blue Text for Metrics */
    [data-testid="stMetricLabel"] {
        color: #0d47a1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1565c0 !important;
    }
    /* Medium Dark Blue Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a237e; /* Medium Dark Blue */
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #ffffff !important;
        border: 1px solid #303f9f;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #283593;
        border-color: #3f51b5;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3949ab !important;
        border-bottom: 3px solid #ff5252;
    }
    /* Expert Interpretation Container */
    .expert-container {
        background-color: #f5f5f5;
        padding: 30px;
        border-radius: 15px;
        border-left: 10px solid #1a237e;
        color: #1a237e;
        margin-top: 50px;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Advanced Structured Evaluation Dashboard")
st.markdown("Challenge 1: RAU Traceability and Deep Metrology")

# Initialize Backend
@st.cache_resource
def init_backend():
    vstore = VectorStoreManager(persist_directory="data_engine/storage/chroma_db")
    pipeline = DualPipeline(vstore)
    metrology = MetrologyEngine()
    interpreter = ExpertInterpreter()
    return pipeline, metrology, interpreter

pipeline, metrology, interpreter = init_backend()

# Sidebar
with st.sidebar:
    st.header("Search & Analysis Params")
    query = st.text_area("Legal Query:", value="Liability and Confidentiality clauses in these contracts")
    k_clusters = st.slider("Clusters (RAU):", 2, 5, 3)
    
    st.divider()
    st.write("Langfuse Monitoring Status")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    if public_key:
        st.success(f"Langfuse Connected: {public_key[:8]}...")
    else:
        st.error("Langfuse Credentials Missing")
    
    run_btn = st.button("Run Parallel Comparison")

# Main Execution
if run_btn:
    with st.spinner("Executing Parallel Synthesis and Deep Evaluation..."):
        # 1. Run Pipelines
        results = pipeline.execute_comparison(query)
        
        # 2. Generate Rich Metrics
        metrics = metrology.generate_rich_metrics(results)
        
        # 3. Dynamic Score Pushing to Langfuse
        pipeline.push_scores_to_langfuse(results["rag"]["trace_id"], {
            "faithfulness": metrics["naive_rag"]["faithfulness"],
            "info_density": metrics["naive_rag"]["info_density"],
            "token_efficiency": metrics["naive_rag"]["efficiency"]
        })
        pipeline.push_scores_to_langfuse(results["rau"]["trace_id"], {
            "faithfulness": metrics["structured_rau"]["faithfulness"],
            "info_density": metrics["structured_rau"]["info_density"],
            "silhouette_score": metrics["structured_rau"]["silhouette"],
            "token_efficiency": metrics["structured_rau"]["efficiency"]
        })

        # 4. Top KPIs Header
        st.subheader("High-Level Performance KPIs")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("RAU Information Density", f"{metrics['structured_rau']['info_density']:.2f}")
        with m_col2:
            st.metric("RAU Faithfulness Score", f"{metrics['structured_rau']['faithfulness']:.1f}/10")
        with m_col3:
            st.metric("RAU Silhouette Quality", f"{metrics['structured_rau']['silhouette']:.3f}")
        with m_col4:
            st.metric("RAU Token Efficiency", f"{metrics['structured_rau']['efficiency']:.1f} t/s")

        st.divider()

        # 5. Result Comparison (Elegant side-by-side)
        res_tab1, res_tab2 = st.tabs(["Synthesized Summaries", "Traceability Center"])
        
        with res_tab1:
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("### Path A: Naive RAG")
                st.write(results["rag"]["summary"])
                st.caption(f"Latency: {metrics['naive_rag']['latency_ms']:.0f}ms | Tokens: {metrics['naive_rag']['total_tokens']}")
                render_trace_button("View RAG Trace in Langfuse", results["rag"].get("trace_url"))
                
            with right_col:
                st.markdown("### Path B: Structured RAU")
                st.write(results["rau"]["summary"])
                st.caption(f"Latency: {metrics['structured_rau']['latency_ms']:.0f}ms | Tokens: {metrics['structured_rau']['total_tokens']}")
                render_trace_button("View RAU Trace in Langfuse", results["rau"].get("trace_url"))

        with res_tab2:
            st.subheader("RAU Chain of Thought Traceability")
            trace_col1, trace_col2 = st.columns([1, 2])
            
            with trace_col1:
                st.markdown("#### Step 1: Cluster Synthesis")
                for i, c_sum in enumerate(results["rau"]["cluster_summaries"]):
                    with st.expander(f"Cluster {i} Theme"):
                        st.write(c_sum)
                
                st.markdown("#### Mathematical Cohesion")
                st.progress(max(0.0, min(1.0, (metrics['structured_rau']['silhouette'] + 1) / 2)), text="Silhouette Score Normalized")
            
            with trace_col2:
                st.markdown("#### Step 2: Semantic Affinity Matrix")
                sim_matrix = results["rau"]["similarity_matrix"]
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(sim_matrix, annot=False, cmap="magma", ax=ax)
                plt.title("Chunk Similarity Heatmap")
                st.pyplot(fig)

        st.divider()

        # 6. Detailed Data Comparison
        st.subheader("Granular Metrology Table")
        comparison_df = pd.DataFrame({
            "Metric": ["Latency (ms)", "Total Tokens", "Info Density", "Faithfulness (1-10)", "Efficiency (t/s)"],
            "Naive RAG": [
                metrics["naive_rag"]["latency_ms"],
                metrics["naive_rag"]["total_tokens"],
                metrics["naive_rag"]["info_density"],
                metrics["naive_rag"]["faithfulness"],
                metrics["naive_rag"]["efficiency"]
            ],
            "Structured RAU": [
                metrics["structured_rau"]["latency_ms"],
                metrics["structured_rau"]["total_tokens"],
                metrics["structured_rau"]["info_density"],
                metrics["structured_rau"]["faithfulness"],
                metrics["structured_rau"]["efficiency"]
            ]
        })
        st.table(comparison_df)

        # 7. Expert Interpreter (Bottom Section)
        st.divider()
        st.subheader("Mission Expert Interpretation")
        with st.spinner("Gemma Analyzing Results..."):
            interpretation = interpreter.interpret_results(metrics, results, trace_id=results["rau"]["trace_id"])
            st.markdown(f"""
            <div class="expert-container">
            {interpretation}
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Initiate a query to see the advanced RAU vs RAG metrology.")

# Zero Icons Verification
st.sidebar.caption("Protocol: Zero Icons Enabled. Plain text UI.")
