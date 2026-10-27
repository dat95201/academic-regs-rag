#!/usr/bin/env bash
# One command from a fresh clone to a running stack:
#   docker check -> .env -> hardware detection -> compose up -> model pull -> ingest -> URL.
# Re-running is safe: images are cached, pulled models are skipped, ingest is incremental.
#
# Modes:
#   ./start.sh                                   local LLM on Ollama (LLM_MODEL, default qwen3:4b)
#   OPENAI_API_KEY=sk-... ./start.sh --cloud     cloud LLM (OPENAI_MODEL, default gpt-4o-mini)
# Embeddings always run on Ollama (bge-m3), so switching mode needs no re-ingest.
set -euo pipefail
cd "$(dirname "$0")"

MODE_ARG=
for arg in "$@"; do
  case "$arg" in
    --cloud|--openai) MODE_ARG=openai ;;
    --local)          MODE_ARG=ollama ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) printf 'Tham số không hợp lệ: %s (dùng --cloud hoặc --local)\n' "$arg" >&2; exit 1 ;;
  esac
done
# Values exported in the shell win over .env (the key is meant to be given at start).
SHELL_LLM_PROVIDER=${LLM_PROVIDER:-}; SHELL_OPENAI_API_KEY=${OPENAI_API_KEY:-}; SHELL_OPENAI_MODEL=${OPENAI_MODEL:-}

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗  %s\033[0m\n' "$*" >&2; exit 1; }

# 1. Docker
command -v docker >/dev/null 2>&1 || die "Chưa cài Docker. Cài Docker Desktop: https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || die "Docker chưa chạy. Mở Docker Desktop (hoặc 'sudo systemctl start docker') rồi chạy lại."
docker compose version >/dev/null 2>&1 || die "Cần Docker Compose v2 ('docker compose')."

# 2. .env
if [ ! -f .env ]; then
  cp .env.example .env
  say "Đã tạo .env từ .env.example"
fi
set -a; # shellcheck disable=SC1091
. ./.env; set +a
[ -n "$SHELL_LLM_PROVIDER" ] && LLM_PROVIDER=$SHELL_LLM_PROVIDER
[ -n "$SHELL_OPENAI_API_KEY" ] && OPENAI_API_KEY=$SHELL_OPENAI_API_KEY
[ -n "$SHELL_OPENAI_MODEL" ] && OPENAI_MODEL=$SHELL_OPENAI_MODEL
[ -n "$MODE_ARG" ] && LLM_PROVIDER=$MODE_ARG
LLM_PROVIDER=$(printf '%s' "${LLM_PROVIDER:-ollama}" | tr '[:upper:]' '[:lower:]')
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
LLM_MODEL=${LLM_MODEL:-qwen3:4b}
EMBED_MODEL=${EMBED_MODEL:-bge-m3}
APP_PORT=${APP_PORT:-8080}
DEFAULT_OLLAMA_URL=http://ollama:11434

# 2b. LLM mode
case "$LLM_PROVIDER" in
  openai)
    [ -n "${OPENAI_API_KEY:-}" ] || die "Chế độ cloud cần OPENAI_API_KEY. Chạy: OPENAI_API_KEY=sk-... ./start.sh --cloud"
    say "Chế độ cloud: LLM = $OPENAI_MODEL (OpenAI); embedding vẫn chạy local ($EMBED_MODEL)"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
      -H "Authorization: Bearer $OPENAI_API_KEY" "https://api.openai.com/v1/models/$OPENAI_MODEL" || true)
    case "$code" in
      200) say "OPENAI_API_KEY hợp lệ" ;;
      401) die "OPENAI_API_KEY không hợp lệ (HTTP 401)." ;;
      403|404) die "Key không dùng được model $OPENAI_MODEL (HTTP $code)." ;;
      *) warn "Không kiểm tra được key với OpenAI (HTTP ${code:-000}), vẫn tiếp tục." ;;
    esac
    ;;
  ollama) say "Chế độ local: LLM = $LLM_MODEL (Ollama)" ;;
  *) die "LLM_PROVIDER không hợp lệ: $LLM_PROVIDER (dùng ollama hoặc openai)" ;;
esac
export LLM_PROVIDER OPENAI_MODEL
export OPENAI_API_KEY=${OPENAI_API_KEY:-}

# 3. Hardware / where Ollama runs
COMPOSE=(docker compose -f docker-compose.yml)
OLLAMA_MODE=container  # container | native | external
if [ -n "${OLLAMA_URL:-}" ] && [ "$OLLAMA_URL" != "$DEFAULT_OLLAMA_URL" ]; then
  OLLAMA_MODE=external
  say "Dùng Ollama tại OLLAMA_URL=$OLLAMA_URL (từ .env)"
else
  case "$(uname -s)" in
    Darwin)
      # Docker on macOS cannot use the Metal GPU; a native Ollama is several times faster.
      if command -v ollama >/dev/null 2>&1; then
        if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
          say "Khởi động Ollama native..."
          (ollama serve >/tmp/ollama-serve.log 2>&1 &)
          for _ in $(seq 1 30); do curl -sf http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
        fi
        curl -sf http://localhost:11434/api/tags >/dev/null || die "Không khởi động được Ollama native (xem /tmp/ollama-serve.log)."
        OLLAMA_MODE=native
        say "macOS: dùng Ollama native (GPU Metal)"
      else
        warn "macOS chưa có Ollama native -> chạy Ollama trong Docker bằng CPU, sẽ CHẬM."
        warn "Nên cài: brew install ollama  (hoặc https://ollama.com/download) rồi chạy lại ./start.sh"
      fi
      ;;
    Linux)
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        COMPOSE+=(-f docker-compose.gpu.yml)
        say "Phát hiện GPU NVIDIA -> bật GPU cho Ollama"
      else
        say "Không có GPU NVIDIA -> chạy CPU (chậm hơn nhưng vẫn dùng được)"
      fi
      ;;
    *) say "Chạy CPU" ;;
  esac
fi

case "$OLLAMA_MODE" in
  container) export OLLAMA_URL=$DEFAULT_OLLAMA_URL; COMPOSE+=(--profile ollama) ;;
  native)    export OLLAMA_URL=http://host.docker.internal:11434 ;;
  external)  export OLLAMA_URL ;;
esac

# 4. Free port for the UI (another stack may already use 8080)
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if [ -z "$("${COMPOSE[@]}" ps -q app 2>/dev/null)" ] && port_busy "$APP_PORT"; then
  NEW_PORT=
  for p in $(seq $((APP_PORT + 1)) $((APP_PORT + 50))); do
    if ! port_busy "$p"; then NEW_PORT=$p; break; fi
  done
  [ -n "$NEW_PORT" ] || die "Cổng $APP_PORT đang bận và không tìm được cổng trống. Đặt APP_PORT trong .env."
  if grep -q '^APP_PORT=' .env; then
    sed -i.bak "s/^APP_PORT=.*/APP_PORT=$NEW_PORT/" .env && rm -f .env.bak
  else
    printf '\nAPP_PORT=%s\n' "$NEW_PORT" >> .env
  fi
  warn "Cổng $APP_PORT đang được chương trình khác dùng -> dùng cổng $NEW_PORT (đã lưu APP_PORT=$NEW_PORT vào .env)"
  APP_PORT=$NEW_PORT
fi
export APP_PORT

# 5. Start services
say "Khởi động services (lần đầu sẽ build image, mất vài phút)..."
"${COMPOSE[@]}" up -d --build --wait

# 6. Models
TTY_FLAG=-T; [ -t 1 ] && TTY_FLAG=
ollama_cli() {
  if [ "$OLLAMA_MODE" = native ]; then ollama "$@"
  else "${COMPOSE[@]}" exec $TTY_FLAG ollama ollama "$@"; fi
}
has_model() {  # works for container/native via CLI, for external via the HTTP API
  local m=$1
  if [ "$OLLAMA_MODE" = external ]; then
    curl -sf "$OLLAMA_URL/api/tags" | grep -Eq "\"name\":\"$m(:latest)?\""
  else
    ollama_cli list | awk 'NR>1 {print $1}' | grep -Eqx "$m(:latest)?"
  fi
}
pull_model() {
  local m=$1
  if has_model "$m"; then say "Model $m đã có"; return; fi
  say "Tải model $m (chỉ lần đầu)..."
  if [ "$OLLAMA_MODE" = external ]; then
    curl -sf "$OLLAMA_URL/api/pull" -d "{\"model\":\"$m\",\"stream\":false}" >/dev/null \
      || die "Không tải được $m từ $OLLAMA_URL"
  else
    ollama_cli pull "$m"
  fi
}
pull_model "$EMBED_MODEL"
[ "$LLM_PROVIDER" = openai ] || pull_model "$LLM_MODEL"

# 7. Ingest (incremental: unchanged documents are skipped)
say "Ingest tài liệu trong data/ ..."
"${COMPOSE[@]}" exec -T app python -m app.ingest.run

# 8. Ready
if curl -sf "http://localhost:$APP_PORT/health" >/dev/null; then
  printf '\n\033[1;32m✅ Mở http://localhost:%s\033[0m  (LLM: %s)\n' "$APP_PORT" \
    "$([ "$LLM_PROVIDER" = openai ] && echo "$OPENAI_MODEL, cloud" || echo "$LLM_MODEL, local")"
else
  warn "Hệ thống chưa sẵn sàng hoàn toàn, xem chi tiết: curl http://localhost:$APP_PORT/health"
fi
echo "   Thử tìm kiếm: curl -G 'http://localhost:$APP_PORT/search' --data-urlencode 'q=thời gian đào tạo tối đa'"
echo "   Dừng: ./stop.sh"
if command -v open >/dev/null 2>&1; then open "http://localhost:$APP_PORT" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:$APP_PORT" >/dev/null 2>&1 || true
fi
