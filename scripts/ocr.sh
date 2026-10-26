#!/usr/bin/env bash
# OCR scanned PDFs in data/raw -> data/processed/<doc_id>.md, inside a throwaway container,
# so the host needs only Docker. Extra args go to the OCR CLI, e.g.:
#   ./scripts/ocr.sh --pages 1-4     # tune on a few pages
#   ./scripts/ocr.sh --check         # report numbering gaps / unreadable margin refs
#   ./scripts/ocr.sh --force         # redo even if the PDF is unchanged
set -euo pipefail
cd "$(dirname "$0")/.."

docker run --rm -v "$PWD":/work -w /work python:3.11-slim bash -c '
  set -e
  apt-get update -qq >/dev/null
  apt-get install -y -qq --no-install-recommends tesseract-ocr tesseract-ocr-vie >/dev/null
  pip install -q --root-user-action=ignore pymupdf pytesseract pillow numpy
  python -m app.ingest.ocr "$@"
' ocr "$@"
