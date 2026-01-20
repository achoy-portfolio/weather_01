#!/bin/bash

# Kill any existing Streamlit processes
pkill -f streamlit

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the betting dashboard
streamlit run dashboard.py
