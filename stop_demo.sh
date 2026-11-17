#!/bin/sh
# Stops the working model. Used by "Stop Phishing Demo.app" on the Desktop.
pkill -f "streamlit run app.py" 2>/dev/null
exit 0
