# Kiến trúc và các quyết định thiết kế

Ghi lại *vì sao* mỗi bước tồn tại, để sau này còn biết cái nào bỏ được.

## 1. Đường đi của dữ liệu

```
data/raw/*.pdf
  │  app/ingest/ocr.py      PDF scan → Markdown (tesseract vie), tách cả chú thích lề
  ▼                          → NGƯỜI RÀ SOÁT (bảng, số điều, ref)
data/processed/*.md
  │  app/ingest/parse.py    PDF có text layer / DOCX / TXT → cùng định dạng Markdown
  │  app/ingest/chunk.py    cắt theo Chương → Điều → khoản, không cắt giữa bảng
  │  app/ingest/run.py      fingerprint + manifest → chỉ embed lại cái đã đổi
  ▼
Qdrant: dense (bge-m3, cosine) + sparse (BM25 tự viết, modifier=IDF)
```

Ingest **idempotent**: `chunk_id` là `uuid5(doc_id|article|index|sha(body))`, manifest lưu
fingerprint từng văn bản. Chạy lại khi không có gì đổi = không làm gì. Đổi `EMBED_MODEL` hoặc
`HYBRID_SEARCH` thì collection bị tạo lại; đổi `CHUNK_MAX_TOKENS`, `CHUNKER_VERSION` hoặc độ
dài trung bình corpus (theo bậc 50 âm tiết) thì re-embed.

## 2. Vì sao tự viết BM25

`fastembed` dùng model `Qdrant/bm25`: stem và bỏ stopword theo tiếng Anh, làm hỏng tiếng Việt.
Từ tiếng Việt chủ yếu 1–3 âm tiết nên `app/rag/embeddings.py` index **âm tiết + bigram âm
tiết** (`"học"`, `"phần"`, `"học phần"`), băm ra chỉ số 31 bit, lưu trọng số term-frequency của
BM25. IDF để Qdrant tự tính (`modifier=IDF`), nên thêm văn bản mới không phải tính lại gì.

`avgdl` lấy từ chính corpus lúc ingest (trước đây hard-code 400) — vì nó vào công thức chuẩn
hoá độ dài, mà chunk của quy chế ngắn hơn giả định ban đầu khá nhiều.

## 3. Truy xuất

```
câu hỏi
 ├─ citations.detect   "Điều 15", "khoản 5.2", "Phụ lục 7"
 ├─ glossary.expand    "GPA" → "điểm trung bình tích luỹ"; "hoc phan" → "học phần"
 ├─ store.search       1 round-trip: query_batch_points([dense, sparse])
 ├─ retriever.fuse     RRF có trọng số, GIỮ nguyên cosine và điểm BM25 của từng nhánh
 ├─ by_article         nếu câu hỏi gọi tên số điều → lấy thẳng, ghép vào danh sách
 ├─ ngưỡng             không có chunk nào đạt MIN_RELEVANCE_SCORE → trả lời "không tìm thấy"
 ├─ rerank.heuristic   chấm lại + phạt trùng Điều
 └─ neighbours         thêm chunk kế bên trong cùng Điều, xếp cuối
```

**Vì sao không để Qdrant tự fuse.** Trước đây dùng `FusionQuery(RRF)` của Qdrant, rồi phải gọi
thêm một lượt `query_points` nữa chỉ để lấy lại cosine thật (vì RRF chỉ còn thứ hạng). Hai
round-trip cho một câu hỏi. Giờ gửi hai nhánh trong **một** request batch rồi fuse ở Python:
bớt một round-trip, và giữ được điểm thật của từng nhánh cho cả ngưỡng lẫn reranker.

**Vì sao không dùng cross-encoder.** `bge-reranker-v2-m3` cần torch + ~2GB, trên máy đã phải
nuôi một LLM và một embedder bằng CPU. Reranker heuristic (`app/rag/rerank.py`) dùng tín hiệu
mà vector search vứt đi:

| Tín hiệu | Trọng số | Vì sao |
|---|---|---|
| điểm fuse (chuẩn hoá theo max) | 1.00 | nền tảng |
| độ phủ từ khoá trong thân chunk | 0.45 | IDF cục bộ: từ xuất hiện ở mọi ứng viên thì vô nghĩa |
| độ phủ trong tiêu đề Điều/Chương | 0.35 | "Điều 28. Phúc tra" trả lời câu hỏi về phúc tra |
| câu hỏi gọi đúng số điều | 1.20 | ý định tường minh thì thắng |
| phạt trùng Điều (luỹ tiến) | −0.18·n(n+1)/2 | đoạn thứ 2 cùng Điều vẫn cạnh tranh, đoạn thứ 4 thì không |

Chuẩn hoá **chia cho max**, không min-max: min-max luôn ánh xạ cao nhất → 1 và thấp nhất → 0,
nên hai ứng viên 0,51 và 0,50 trông xa nhau như 1,0 và 0,0 và không tín hiệu nào lay chuyển
được nữa.

`RERANK_MODE=llm` là lựa chọn khác: gửi *tiêu đề* các ứng viên cho chính model đang chạy và
xin thứ tự. Tốn một round-trip, hợp với câu hỏi mơ hồ; parse lỗi thì tự quay về thứ tự cũ.

**Mở rộng ngữ cảnh.** Câu trả lời của quy chế hay nằm vắt giữa 15.3 và 15.4. Sau khi chốt
top-k, lấy thêm chunk kế bên trong cùng Điều, **xếp cuối** — `build_context` cắt theo ngân
sách token từ cuối lên, nên phần bổ sung là thứ bị bỏ đầu tiên khi chật chỗ.

## 4. Sinh câu trả lời

Prompt (`app/rag/prompt.py`) ràng buộc: chỉ dùng trích đoạn, mỗi ý một trích dẫn, giữ nguyên
con số, nêu rõ từng đối tượng nếu quy định khác nhau. Trích đoạn được đánh số và kèm tiêu đề
đầy đủ, nhờ đó model trích dẫn được tên Điều chứ không phải "[1]".

**Rewrite câu hỏi nối tiếp** chỉ chạy khi cần (`chat.needs_rewrite`): có đại từ/liên từ nối
("còn...", "vậy...", "trường hợp này") hoặc quá ngắn. Trước đây *mọi* câu hỏi nối tiếp đều tốn
một lượt gọi LLM trước khi tìm kiếm.

**Soát trích dẫn** (`app/rag/citations.py`): sau khi model viết xong, mọi `[..., Điều N]` được
đối chiếu với các Điều thật sự có trong context. Cái nào không có thì `done.unsupported_citations`
báo về và giao diện đánh dấu — không sửa câu trả lời, chỉ không để con số bịa trông đáng tin.

## 5. Luồng sự kiện SSE

```
query?      câu hỏi đã viết lại (chỉ khi có rewrite)
retrieved   TRÍCH ĐOẠN — gửi ngay khi tìm xong, TRƯỚC token đầu tiên
token*      câu trả lời, từng mẩu
sources     trích đoạn đã dùng ([] nếu model bảo không có thông tin)
done        found, độ trễ, trích dẫn đáng ngờ, gợi ý khi không tìm thấy
```

Trên máy CPU, `retrieved` tới trước token đầu vài giây. Người đọc mở trích đoạn gốc được ngay
trong lúc model còn đang viết — đó là lý do tách sự kiện này ra.

## 6. Quan sát và đo đạc

- `data/logs/queries.jsonl`: mỗi lượt một dòng (câu hỏi, câu đã rewrite, số ứng viên, độ trễ,
  tìm thấy hay không, trích dẫn đáng ngờ). `GET /metrics` tổng hợp lại.
- `data/logs/feedback.jsonl`: 👍/👎 từ giao diện.
- `/debug`: xem điểm từng nhánh của từng chunk cho một câu hỏi bất kỳ.
- `app/eval/`: bộ câu hỏi vàng + harness đo recall@k, hit@final, MRR. `--compare` bật dần từng
  kỹ thuật để thấy đóng góp thật của mỗi cái.

Ghi log là cục bộ, không gửi đi đâu; tắt bằng `LOG_QUERIES=false`.

## 7. Còn nợ

- Hiện `/chat` là endpoint đồng bộ chạy trong threadpool của FastAPI. Với vài chục người dùng
  đồng thời nên chuyển `httpx.AsyncClient` + `async def` để không chiếm thread.
- Chưa có xác thực cho `/ingest`: ai vào được cổng là kích được ingest. Chấp nhận được khi
  chạy localhost, cần khoá lại nếu deploy.
- Chưa có cache câu trả lời theo ngữ nghĩa (mới cache vector của câu hỏi).
- OCR vẫn cần người rà soát; chưa tự so khớp số Điều giữa bản OCR và mục lục gốc.
