#!/bin/bash
PORT=5006
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/strategy_app.log"
PID=$(ss -tlnp | grep ":${PORT}" | grep -oP 'pid=\K[0-9]+')
if [ -n "$PID" ]; then
    echo "Killing old process PID=$PID on port $PORT..."
    kill -9 "$PID" 2>/dev/null
    sleep 1
fi
cd "$APP_DIR"
nohup python3 src/app.py >> "$LOG" 2>&1 &
echo "Started PID=$!"
sleep 2
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/ 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Service running on port $PORT"
else
    echo "❌ Failed (HTTP $HTTP_CODE)"
    tail -10 "$LOG"
fi
