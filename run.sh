#!/bin/bash
set -e
cd "$(dirname "$0")/backend"
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
rm -f revenue_recovery.db
echo "Starting backend on http://localhost:8000 ..."
echo "Open frontend/index.html in your browser once this is running."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
