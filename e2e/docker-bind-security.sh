#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

base_image=; head_image=; fixture=; base_ref="${HEADROOM_BASE_REF:-origin/main}"
while (($#)); do
  case "$1" in
    --base-image) base_image="$2"; shift 2 ;;
    --head-image) head_image="$2"; shift 2 ;;
    --fixture) fixture="$2"; shift 2 ;;
    --base-ref) base_ref="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$base_image" && -n "$head_image" && -n "$fixture" ]] || exit 2
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "python3 or python is required" >&2; exit 127; }

base_container="headroom-bind-base-$$"; head_container="headroom-bind-head-$$"
local_container="headroom-bind-local-$$"; public_container="headroom-bind-public-$$"
attacker_container="headroom-bind-attacker-$$"; mock_container="headroom-bind-upstream-$$"
slim_container="headroom-bind-slim-$$"; network="headroom-bind-network-$$"
tmp_dir="$(mktemp -d)"
cleanup() {
  docker rm -f "$attacker_container" "$public_container" "$local_container" "$slim_container" "$head_container" "$base_container" "$mock_container" >/dev/null 2>&1 || true
  # The CI workflow reuses the head image for installer, compose, and wrap checks.
  docker image rm "$base_image" "${head_image}-runtime-slim" >/dev/null 2>&1 || true
  docker network rm "$network" >/dev/null 2>&1 || true
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mapfile -t request_paths < <("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
for path in json.load(open(sys.argv[1], encoding="utf-8"))["request_paths"]:
    print(path)
PY
)
upstream_path="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["upstream"]["path"])
PY
)"
missing_status="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["token_control"]["missing"])
PY
)"
wrong_status="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["token_control"]["wrong"])
PY
)"
correct_status="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["token_control"]["correct"])
PY
)"
ws_status_expected="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["websocket_control"]["missing"])
PY
)"
upstream_path="${upstream_path//$'\r'/}"

fixture_description="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["description"])
PY
)"
upstream_method="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["upstream"]["method"])
PY
)"
upstream_authorization="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["upstream"]["authorization"]
print("" if value is None else value)
PY
)"
network_publication="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["network"]["publication"])
PY
)"
network_peer="$("$PYTHON_BIN" - "$fixture" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["network"]["peer"])
PY
)"

test "$fixture_description" = "Unauthenticated public Docker publication"
test "$upstream_method" = POST
test -z "$upstream_authorization"
test "$network_publication" = "0.0.0.0:8787:8787"
test "$network_peer" = non-loopback
git archive "$base_ref" | tar -x -C "$tmp_dir"
docker build -t "$base_image" "$tmp_dir" >/dev/null
docker build -t "$head_image" . >/dev/null
slim_image="${head_image}-runtime-slim"
docker build --target runtime-slim -t "$slim_image" . >/dev/null
docker network create "$network" >/dev/null
mock_script='from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
    def _reply(self):
        print(self.command + " " + self.path + " authorization=" + str(self.headers.get("authorization")), flush=True)
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b"{}")
    do_GET = _reply
    do_POST = _reply
    def log_message(self, *_): pass
HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()'
docker run -d --rm --name "$mock_container" --network "$network" python:3.12-alpine python -c "$mock_script" >/dev/null
docker run -d --rm --name "$base_container" --network "$network" -p "0.0.0.0:18783:8787" \
  -e "OPENAI_TARGET_API_URL=http://${mock_container}:8080" -e OPENAI_API_KEY=upstream-secret \
  "$base_image" >/dev/null
docker run -d --rm --name "$head_container" --network "$network" -p "0.0.0.0:18784:8787" "$head_image" >/dev/null
docker run -d --rm --name "$local_container" --network "$network" -p "127.0.0.1:18786:8787" \
  -e "OPENAI_TARGET_API_URL=http://${mock_container}:8080" -e OPENAI_API_KEY=upstream-secret \
  "$head_image" --host 0.0.0.0 --port 8787 >/dev/null
docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$slim_image" \
  | grep -Fxq 'HEADROOM_HOST=127.0.0.1'
docker run --rm --entrypoint python3 "$slim_image" -c 'import headroom._core; print("rust_core=loaded")'

host_ip="$(docker network inspect -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' "$network")"
wait_healthy() {
  local container="$1"
  for _ in $(seq 1 60); do
    if docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null | grep -Fxq healthy; then
      return 0
    fi
    sleep 1
  done
  docker inspect -f '{{.State.Health.Status}}' "$container"
  docker logs "$container" 2>&1 || true
  return 1
}
wait_healthy "$base_container"
wait_healthy "$head_container"
wait_healthy "$local_container"
docker exec "$base_container" python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8787/readyz", timeout=5)'
docker exec "$head_container" python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8787/readyz", timeout=5)'
docker exec "$local_container" python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8787/readyz", timeout=5)'
request() {
  local port="$1" path="$2" token="${3:-}"; local -a args=(--silent --output /dev/null --write-out '%{http_code}' --connect-timeout 3)
  [[ -n "$token" ]] && args+=(-H "Authorization: Bearer ${token}")
  [[ "$path" == "$upstream_path" ]] && args+=(-H 'Content-Type: application/json' --data '{"model":"fixture","messages":[]}')
  docker run --rm --name "$attacker_container" --network "$network" curlimages/curl:8.12.1 \
    "${args[@]}" "http://${host_ip}:${port}${path}"
}
first_path="${request_paths[0]//$'\r'/}"
base_first_status="$(request 18783 "$first_path" || true)"
if [[ "$base_first_status" == 200 ]]; then
  # The vulnerable base must retain the issue-shaped unauthenticated exposure.
  for path in "${request_paths[@]}"; do
    path="${path//$'\r'/}"
    test "$(request 18783 "$path")" = 200
  done
  docker logs "$mock_container" 2>&1 | grep -Fq "${upstream_method} ${upstream_path} authorization=${upstream_authorization:-None}"
elif [[ "$base_first_status" != 000 && "$base_first_status" != 401 && "$base_first_status" != 403 ]]; then
  echo "unexpected safe-base status: $base_first_status" >&2
  exit 1
fi
test "$(request 18784 "$first_path")" = 000
test "$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 3 "http://127.0.0.1:18786${first_path}")" = 200
test "$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 3 --header 'Content-Type: application/json' --data '{"model":"fixture","messages":[]}' "http://127.0.0.1:18786${upstream_path}")" = 200
test "$(request 18786 "$first_path" || true)" = 000
token="$(openssl rand -hex 32)"
docker run -d --rm --name "$public_container" --network "$network" -p 0.0.0.0:18785:8787 \
  -e "HEADROOM_PROXY_TOKEN=${token}" -e "OPENAI_TARGET_API_URL=http://${mock_container}:8080" \
  -e OPENAI_API_KEY=upstream-secret \
  "$head_image" --host 0.0.0.0 --port 8787 >/dev/null
for _ in $(seq 1 60); do curl -4 --fail --silent http://127.0.0.1:18785/readyz >/dev/null && break; sleep 1; done
for path in "${request_paths[@]}"; do
  path="${path//$'\r'/}"
  test "$(request 18785 "$path")" = "$missing_status"
  test "$(request 18785 "$path" wrong-token)" = "$wrong_status"
  test "$(request 18785 "$path" "$token")" = "$correct_status"
done
ws_key="$(printf '%s' 'the sample nonce' | base64 | tr -d '\n')"
ws_status="$(docker run --rm --name "$attacker_container" --network "$network" curlimages/curl:8.12.1 --silent --output /dev/null --write-out '%{http_code}' --http1.1 -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' -H "Sec-WebSocket-Key: ${ws_key}" "http://${host_ip}:18785/v1/live")"
test "$ws_status" = "$ws_status_expected"
printf 'base_ref=%s base_digest=%s head_digest=%s slim_digest=%s runtime_slim_config=loopback rust_core=loaded local_completion=200 attacker_loopback=000 ws_status=%s\n' "$base_ref" "$(docker image inspect --format='{{.Id}}' "$base_image")" "$(docker image inspect --format='{{.Id}}' "$head_image")" "$(docker image inspect --format='{{.Id}}' "$slim_image")" "$ws_status"
echo "docker bind security: fixture forwarding, loopback isolation, built-image health, slim-image loopback config, and token controls passed"
