#!/bin/bash
echo "\ud83e\uddea Testando Terminal AI V2..."
source venv/bin/activate

echo ""
echo "1. Testando /api/health..."
curl -s http://localhost:8001/api/health | python3 -m json.tool

echo ""
echo "2. Verificando sandbox..."
ls -la sandbox/

echo ""
echo "\u2713 Testes conclu\u00eddos!"
