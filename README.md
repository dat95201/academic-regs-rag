# Hỏi đáp quy chế học vụ (RAG, chạy local)

Chatbot tra cứu *Quy định về học vụ và đào tạo bậc đại học* — trả lời bằng tiếng Việt, mỗi ý
kèm trích dẫn **Điều / Phụ lục** bấm được để mở đúng đoạn văn bản gốc và đúng trang PDF.

Toàn bộ chạy trên máy bạn: Ollama (LLM + embedding) + Qdrant (vector store) + FastAPI.
Không gửi dữ liệu đi đâu, trừ khi bạn chủ động bật chế độ cloud.

```bash
./start.sh                                  # LLM local (qwen3:4b)
OPENAI_API_KEY=sk-... ./start.sh --cloud    # LLM cloud, embedding vẫn local
```

Lần đầu mất vài phút để build image và tải model. Xong thì mở http://localhost:8080.

| Trang | Dùng để |
|---|---|
| `/` | Hỏi đáp, có trích dẫn bấm được, mở PDF đúng trang, lưu hội thoại trong trình duyệt |
| `/browse` | Đọc toàn văn theo Chương → Điều → Phụ lục, có ô lọc nhanh |
| `/debug` | Xem bộ truy xuất trả về gì: điểm từng nhánh, từ được mở rộng, chunk bị loại |

## Hoạt động thế nào

```
data/raw/*.pdf ──OCR──► data/processed/*.md ──chunk──► embedding ──► Qdrant
                         (người rà soát lại)           (bge-m3)      (dense + BM25)

câu hỏi ─► mở rộng thuật ngữ ─► tìm dense + BM25 (1 round-trip) ─► fuse ─► rerank
        ─► mở rộng ngữ cảnh ─► prompt kèm trích đoạn ─► LLM ─► câu trả lời + nguồn
```

Chi tiết từng bước và lý do chọn: [docs/architecture.md](docs/architecture.md).

Vài điểm đáng chú ý:

- **Chunk theo cấu trúc văn bản.** Một Điều = một chunk nếu vừa ngưỡng token; quá dài thì cắt
  theo khoản, không cắt giữa bảng. Mỗi chunk mang theo đường dẫn ngữ cảnh
  `[Quy định học vụ ĐH > Chương 2 > Điều 5. Thời gian đào tạo > 5.2]` khi đem đi embed.
- **Hybrid search.** Dense (`bge-m3`) bắt câu hỏi diễn đạt khác; BM25 tự viết cho tiếng Việt
  (âm tiết + bigram âm tiết, không stem kiểu tiếng Anh) bắt đúng con chữ.
- **Mở rộng thuật ngữ.** Sinh viên hỏi "GPA", "rớt môn", "đuổi học", "hoc phan" — văn bản viết
  "điểm trung bình tích luỹ", "học phần không đạt", "buộc thôi học". Bảng tra trong
  `app/rag/glossary.py`, không cần model.
- **Tra thẳng theo số điều.** "Điều 15 nói gì?" lấy đúng Điều 15 từ Qdrant thay vì trông chờ
  vector đồng ý.
- **Rerank không cần cross-encoder.** Chấm lại ứng viên bằng độ phủ từ khoá, trùng tiêu đề,
  trùng số điều, cộng phạt trùng lặp để không nhét 5 đoạn của cùng một Điều.
- **Soát trích dẫn.** Nếu câu trả lời trích một Điều không có trong trích đoạn đã dùng, giao
  diện cảnh báo thay vì để con số đó trông có vẻ đáng tin.

## Thêm tài liệu

1. Bỏ file vào `data/raw/` (PDF, DOCX, MD, TXT).
2. PDF scan: `./scripts/ocr.sh` rồi **rà soát tay** `data/processed/<doc_id>.md` — nhất là bảng
   và chú thích lề `<!-- ref: ... -->`. `./scripts/ocr.sh --check` báo chỗ thiếu số Điều.
3. `make ingest` (hoặc chạy lại `./start.sh`). Ingest là incremental: file không đổi thì bỏ qua.

## Công cụ

```bash
make help          # danh sách lệnh
make test          # test trong container tạm, không cần Ollama/Qdrant
make eval          # đo recall@k / MRR trên bộ câu hỏi vàng
make eval-compare  # đo từng kỹ thuật một, so với baseline dense-only
make metrics       # tỉ lệ trả lời được, độ trễ, câu hỏi không ai trả lời được
make lint
```

`make eval-compare` bật thêm một kỹ thuật mỗi dòng và in ra bảng dạng dưới đây (số là minh hoạ
định dạng, **chạy trên máy bạn mới ra số thật**). Dùng nó để quyết định giữ hay bỏ một kỹ
thuật, đừng đoán:

```
                     recall  hit@k    MRR   ms
baseline (dense only)  ...     ...     ...   ...
+ hybrid BM25          ...     ...     ...   ...
+ glossary             ...     ...     ...   ...
+ article lookup       ...     ...     ...   ...
+ rerank (current)     ...     ...     ...   ...
```

Bộ câu hỏi vàng ở `app/eval/questions.toml`. Mỗi khi hệ thống trả lời sai một câu, thêm câu đó
vào file — `/metrics` và `data/logs/queries.jsonl` cho biết người dùng thật sự hỏi gì.

## Cấu hình

Mọi thứ nằm trong `.env` (xem `.env.example` có chú thích từng khoá). Vài khoá hay chỉnh:

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `LLM_MODEL` | `qwen3:4b` | Model trả lời khi chạy local |
| `TOP_K_RETRIEVE` / `TOP_K_FINAL` | 20 / 5 | Số ứng viên lấy về / **trần** số trích đoạn đưa vào prompt |
| `RELATIVE_CUTOFF` | 0.85 | Chỉ giữ chunk còn trong khoảng này so với chunk tốt nhất. Đây là thứ làm số nguồn thay đổi theo câu hỏi thay vì luôn bằng `TOP_K_FINAL` |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | 1.0 / 0.8 | Trọng số hai nhánh khi fuse |
| `RERANK_MODE` | `heuristic` | `heuristic` (miễn phí) \| `llm` (thêm 1 lượt gọi) \| `off` |
| `MIN_RELEVANCE_SCORE` | 0.3 | Dưới ngưỡng này thì trả lời "không tìm thấy" |
| `CONTEXT_NEIGHBOURS` | 2 | Số chunk lân cận lấy thêm làm ngữ cảnh |

Đổi `CHUNK_MAX_TOKENS`, `EMBED_MODEL` hay `HYBRID_SEARCH` sẽ tự động ingest lại.

## Bố cục

```
app/
  api/      main.py (HTTP), library.py (đọc văn bản), telemetry.py (log truy vấn)
  ingest/   ocr.py → parse.py → chunk.py → run.py
  rag/      glossary.py → retriever.py → rerank.py → prompt.py → llm.py → chat.py
            store.py (Qdrant), embeddings.py (dense + BM25), citations.py (soát trích dẫn)
  eval/     run.py + questions.toml (bộ câu hỏi vàng)
  static/   index.html (chat), browse.html (tra cứu), debug.html (chẩn đoán), style.css
tests/      pytest, chạy được không cần Ollama/Qdrant
```

## Khi có trục trặc

| Hiện tượng | Xử lý |
|---|---|
| Thanh trạng thái báo "Chưa kết nối được Ollama" | `docker compose ps`; trên macOS nên `brew install ollama` để dùng GPU Metal |
| "Chưa có tài liệu (chưa ingest)" | `make ingest` |
| Trả lời chậm | Dùng `./start.sh --cloud`, hoặc model nhỏ hơn, hoặc giảm `TOP_K_FINAL` |
| Trả lời sai điều/khoản | Mở `/debug` với đúng câu hỏi đó, xem chunk nào được lấy; thêm từ vào `glossary.py` hoặc thêm câu vào bộ eval |
| Câu trả lời trích điều không tồn tại | Giao diện đã cảnh báo; thường do model nhỏ — thử `--cloud` để đối chiếu |
