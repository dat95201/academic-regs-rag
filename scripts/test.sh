#!/usr/bin/env bash
# Run the test suite in a throwaway container, so the host needs only Docker.
# The tests need neither Ollama nor Qdrant: they use an in-memory Qdrant and a fake embedder.
#   ./scripts/test.sh                # everything
#   ./scripts/test.sh tests/test_retriever.py -k article
set -euo pipefail
cd "$(dirname "$0")/.."

docker run --rm -v "$PWD":/work -w /work python:3.11-slim bash -c '
  set -e
  pip install -q --root-user-action=ignore \
    pydantic-settings fastapi httpx "qdrant-client>=1.15,<1.16" pytest
  python -m pytest "$@"
' pytest "${@:-tests -q}"
