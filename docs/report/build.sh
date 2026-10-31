#!/usr/bin/env bash
# Biên dịch báo cáo ra PDF. Bắt buộc XeLaTeX: báo cáo dùng tiếng Việt Unicode và
# font hệ thống (TeX Gyre Pagella, Carlito, DejaVu Sans Mono).
#
#   ./build.sh            dùng xelatex trên máy, nếu không có thì dùng Docker
#   ./build.sh --docker   ép dùng Docker
#   ./build.sh --clean    xoá file trung gian
set -euo pipefail
cd "$(dirname "$0")"

OUT=BaoCao-RAG-QuyDinhHocVu-CO5179.pdf

if [ "${1:-}" = "--clean" ]; then
  rm -f main.{aux,log,out,toc,lof,lot,fls,fdb_latexmk} *.aux
  echo "Đã xoá file trung gian."; exit 0
fi

run_local() {
  # Chạy hai lượt: lượt đầu dựng mục lục và tham chiếu, lượt sau điền số trang.
  for i in 1 2; do
    xelatex -interaction=nonstopmode -file-line-error main.tex > /tmp/xelatex-$i.log 2>&1 || {
      echo "Lỗi biên dịch, xem /tmp/xelatex-$i.log:"; grep -E "^.*:[0-9]+:" /tmp/xelatex-$i.log | head -20; exit 1; }
  done
}

run_docker() {
  docker run --rm -v "$PWD":/w -w /w texlive/texlive:latest-small \
    bash -c 'for i in 1 2; do xelatex -interaction=nonstopmode main.tex >/dev/null; done'
}

if [ "${1:-}" = "--docker" ]; then
  command -v docker >/dev/null || { echo "Chưa cài Docker."; exit 1; }
  echo "Biên dịch bằng Docker (lần đầu tải image, vài phút)..."; run_docker
elif command -v xelatex >/dev/null 2>&1; then
  echo "Biên dịch bằng xelatex trên máy..."; run_local
elif command -v docker >/dev/null 2>&1; then
  echo "Máy chưa có xelatex -> dùng Docker (lần đầu tải image, vài phút)..."; run_docker
else
  echo "Cần XeLaTeX (macOS: brew install --cask mactex-no-gui) hoặc Docker." >&2; exit 1
fi

mv -f main.pdf "$OUT"
echo "✅ $OUT  ($(command -v pdfinfo >/dev/null && pdfinfo "$OUT" | awk '/Pages/{print $2" trang"}' || echo 'xong'))"
