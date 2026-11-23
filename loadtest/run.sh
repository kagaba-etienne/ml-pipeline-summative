#!/usr/bin/env bash
# Run the same Locust test against 1, 2 and 4 API containers. Run from the repo root on the
# machine running docker compose. Results: loadtest/api<N>_stats.csv and loadtest/api<N>.html
#
#   HTTP_PORT=8080 loadtest/run.sh          # local (app on :8080)
#   loadtest/run.sh                         # VPS (app on :80)
set -euo pipefail
PORT=${HTTP_PORT:-80}
USERS=${USERS:-30}
DURATION=${DURATION:-2m}

for n in 1 2 4; do
  echo "== $n API container(s)"
  docker compose up -d --scale api=$n >/dev/null 2>&1
  until [ "$(docker compose ps api --format '{{.Status}}' | grep -c '(healthy)')" = "$n" ]; do sleep 2; done
  sleep 8   # let nginx pick up the new containers
  uv run locust --host "http://localhost:$PORT" --headless -u "$USERS" -r 3 -t "$DURATION" \
      --only-summary --csv "loadtest/api$n" --html "loadtest/api$n.html" > /dev/null 2>&1
  tail -1 "loadtest/api${n}_stats.csv"
done
docker compose up -d --scale api=1 >/dev/null 2>&1
