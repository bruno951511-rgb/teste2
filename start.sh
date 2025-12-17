#!/bin/bash
echo "\ud83d\ude80 Iniciando Terminal AI V2..."
source venv/bin/activate
uvicorn serve_v2_complete:app --host 0.0.0.0 --port 8001 --reload
