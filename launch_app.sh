#!/bin/bash
# Mission: Challenge 1 Launcher
# Environment: Python 3.13.12 (venv)

# 1. Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv directory not found. Please run environment setup first."
    exit 1
fi

# 2. Set PYTHONPATH to include data_engine and challenges
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Launch Streamlit Dashboard
echo "Starting Streamlit Dashboard on port 8501..."
streamlit run ui/dashboard.py --server.port 8501 --server.headless true
