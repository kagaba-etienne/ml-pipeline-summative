#!/usr/bin/env bash
# Run the same Locust test against 1, 2 and 4 API containers. Run from the repo root.
# Results: loadtest/<PREFIX><N>_stats.csv and loadtest/<PREFIX><N>.html
#
#   HTTP_PORT=8080 loadtest/run.sh                                   # local docker (app on :8080)
#   SSH=user@192.168.1.1 HOST=http://192.168.1.1 PREFIX=vps_api loadtest/run.sh
#                                                                    # VPS: Locust runs here, containers
#                                                                    # are scaled there over ssh
set -euo pipefail
HOST=${HOST:-http://localhost:${HTTP_PORT:-80}}
SSH=${SSH:-}                    # user@server running docker compose (empty = this machine)
APP_DIR=${APP_DIR:-ml-pipeline-summative}   # on the server, relative to the home folder
PREFIX=${PREFIX:-api}
USERS=${USERS:-30}
DURATION=${DURATION:-2m}

compose() {
  if [ -n "$SSH" ]; then ssh "$SSH" "cd $APP_DIR && docker compose $(printf '%q ' "$@")"
  else docker compose "$@"; fi
}

for n in 1 2 4; do
  echo "== $n API container(s)"
  compose up -d --scale api=$n 2>&1 | tail -1     # shows the error if it fails
  until [ "$(compose ps api --format '{{.Status}}' | grep -c '(healthy)')" = "$n" ]; do sleep 3; done
  sleep 8   # let nginx pick up the new containers
  uv run locust --host "$HOST" --headless -u "$USERS" -r 3 -t "$DURATION" \
      --only-summary --csv "loadtest/$PREFIX$n" --html "loadtest/$PREFIX$n.html" > /dev/null 2>&1 || echo "(some requests failed, see the report)"
  tail -1 "loadtest/$PREFIX${n}_stats.csv"
done
compose up -d --scale api=1 >/dev/null 2>&1
