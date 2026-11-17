#!/bin/sh
# Starts the working model in the background on http://localhost:8501 and opens
# it in the browser. Used by "Start Phishing Demo.app" on the Desktop.
cd "$(dirname "$0")" || exit 1
pkill -f "streamlit run app.py" 2>/dev/null
sleep 1
nohup .venv/bin/streamlit run app.py --server.headless true --server.port 8501 \
  > results/app.log 2>&1 < /dev/null &
for i in $(seq 1 40); do
  curl -s -o /dev/null http://localhost:8501 && break
  sleep 1
done
open "http://localhost:8501"
