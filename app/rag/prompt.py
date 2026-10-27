"""Prompts (Vietnamese) and context assembly."""

from __future__ import annotations

from app.ingest.chunk import estimate_tokens

NOT_FOUND = "Tài liệu hiện có không đề cập vấn đề này."

SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu quy chế học vụ cho sinh viên. Trả lời bằng tiếng Việt.

Quy tắc bắt buộc:
1. Chỉ dùng thông tin trong các TRÍCH ĐOẠN được cung cấp. Không dùng kiến thức bên ngoài, không suy đoán.
2. Nếu các trích đoạn không chứa thông tin để trả lời, trả lời đúng một câu: "{NOT_FOUND}"
3. Mỗi ý trong câu trả lời phải kèm trích dẫn dạng [Tên văn bản, Điều X] hoặc [Tên văn bản, Phụ lục X], lấy đúng tên văn bản và Điều/Phụ lục ghi ở đầu trích đoạn. Có thể ghi thêm mục, ví dụ [Quy định học vụ ĐH (hợp nhất 2022), Điều 5.2].
4. Giữ nguyên con số, mốc thời gian, điều kiện như trong trích đoạn. Nếu quy định khác nhau theo đối tượng (chính quy, vừa làm vừa học, khoá tuyển sinh...), nêu rõ từng trường hợp.
5. Trả lời ngắn gọn, đi thẳng vào câu hỏi; dùng gạch đầu dòng khi có nhiều ý.
6. Không thêm dòng tóm tắt kiểu "→ Đáp án: ..." hay "Đáp án: ..." ở cuối; câu trả lời phía trên đã là đáp án."""

REWRITE_PROMPT = """Dựa vào hội thoại trước đó, viết lại câu hỏi cuối của người dùng thành một câu hỏi độc lập, đầy đủ ý, để tra cứu trong quy chế học vụ. Giữ nguyên thuật ngữ. Nếu câu hỏi đã đủ ý thì giữ nguyên. Chỉ trả về câu hỏi, không giải thích."""


def label(payload: dict) -> str:
    """Human label of a chunk: 'Điều 5 (5.1–5.3)' / 'Phụ lục 9'."""
    where = payload.get("article") or "Phần mở đầu"
    sections = payload.get("sections") or []
    if len(sections) == 1:
        where += f" (mục {sections[0]})"
    elif sections:
        where += f" (mục {sections[0]}–{sections[-1]})"
    return where


def build_context(payloads: list[dict], max_tokens: int) -> tuple[str, list[dict]]:
    """Numbered excerpts within the token budget; returns (context, payloads used)."""
    parts, used, total = [], [], 0
    for p in payloads:
        refs = ", ".join(p.get("source_refs") or [])
        head = (
            f"[{len(used) + 1}] {p.get('doc_short_title') or p.get('doc_title')} – {label(p)}"
            + (f" – {p.get('article_title')}" if p.get("article_title") else "")
            + f" – trang {p.get('page')}"
            + (f" (nguồn gốc: {refs})" if refs else "")
        )
        block = f"{head}\n{p.get('text', '')}"
        cost = estimate_tokens(block)
        if used and total + cost > max_tokens:
            break
        parts.append(block)
        used.append(p)
        total += cost
    return "\n\n---\n\n".join(parts), used


def answer_messages(question: str, context: str, history: list[dict]) -> list[dict]:
    user = f"TRÍCH ĐOẠN:\n\n{context}\n\n---\n\nCÂU HỎI: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user}]


def rewrite_messages(question: str, history: list[dict]) -> list[dict]:
    convo = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:600]}" for m in history
    )
    return [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Hội thoại:\n{convo}\n\nCâu hỏi cuối: {question}"},
    ]
