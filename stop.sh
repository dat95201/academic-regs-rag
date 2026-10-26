#!/usr/bin/env bash
# Stop all services. Volumes (pulled models, Qdrant data) are kept.
# To wipe everything: docker compose --profile ollama down -v
set -euo pipefail
cd "$(dirname "$0")"
docker compose -f docker-compose.yml --profile ollama down
echo "Đã dừng. Model và dữ liệu Qdrant vẫn được giữ trong Docker volume."
