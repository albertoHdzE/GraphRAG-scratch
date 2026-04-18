# AGENTS.md

## Running the App
```bash
./launch_app.sh  # Starts Streamlit on port 8501
```
- Requires `.venv` (Python 3.13 venv already present)
- Sets `PYTHONPATH=$PYTHONPATH:$(pwd)` automatically

## Key Commands
```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
streamlit run ui/dashboard.py --server.port 8501

# Verification (e2e test)
python verify_mission.py
```

## Architecture
- **Entry**: `ui/dashboard.py` (Streamlit)
- **Data Engine**: `data_engine/` (processor.py, vector_store.py, ingest.py)
- **Pipeline**: `challenges/c01_clustering_rau/` (pipeline.py, comparison.py)
- **Vector Store**: ChromaDB at `data_engine/storage/chroma_db/`
- **LLM**: Gemma-3-4b via LM Studio on `localhost:1234`
- **Observability**: LangFuse Cloud

## Zero Icons Policy
- No emojis or Unicode symbols anywhere in code, logs, comments, or reports
- Use alphanumeric only in all strings

## Env Requirements
- `.env` must contain LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
- LangFuse SDK sends traces to cloud.langfuse.com