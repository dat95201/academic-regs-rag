"""Query expansion: student wording -> the wording the regulation actually uses.

Students ask with abbreviations and campus slang ("GPA bao nhiêu thì bị đuổi học?",
"rớt môn thì sao?", "bao giờ được bảo lưu?"), while the document spells everything out
("điểm trung bình tích luỹ", "buộc thôi học", "học phần không đạt", "nghỉ học tạm thời").
Dense search tolerates some of that; BM25 does not, and BM25 is what pins down the exact
article. So the query is expanded before searching:

    expand("GPA thấp có bị đuổi học không?").text
    -> "GPA thấp có bị đuổi học không? — điểm trung bình tích luỹ; buộc thôi học"

Also handles queries typed without diacritics ("hoc phan", "canh bao hoc tap"), which would
otherwise match nothing at all in a Vietnamese BM25 index.

No model and no network: a table plus word-boundary matching. Grow the table from
data/logs/queries.jsonl, which records what people actually type.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Abbreviations students use. Matched as whole words, case-insensitively for 3+ characters;
# 2-character ones must be upper-case so "tc"/"hk" inside ordinary text is left alone.
ACRONYMS: dict[str, str] = {
    "GPA": "điểm trung bình tích luỹ",
    "CPA": "điểm trung bình tích luỹ",
    "DTB": "điểm trung bình",
    "ĐTB": "điểm trung bình",
    "ĐTBTL": "điểm trung bình tích luỹ",
    "ĐTBHK": "điểm trung bình học kỳ",
    "CTĐT": "chương trình đào tạo",
    "ĐCMH": "đề cương chi tiết học phần",
    "HP": "học phần",
    "TC": "tín chỉ",
    "SV": "sinh viên",
    "GV": "giảng viên",
    "HK": "học kỳ",
    "TKB": "thời khoá biểu",
    "ĐKHP": "đăng ký học phần",
    "ĐATN": "học phần tốt nghiệp đồ án tốt nghiệp",
    "LVTN": "học phần tốt nghiệp luận văn tốt nghiệp",
    "KLTN": "học phần tốt nghiệp khoá luận tốt nghiệp",
    "BTL": "bài tập lớn",
    "GDQP": "Giáo dục Quốc phòng - An ninh",
    "GDQPAN": "Giáo dục Quốc phòng - An ninh",
    "GDTC": "Giáo dục Thể chất",
    "CLC": "chất lượng cao",
    "VLVH": "vừa làm vừa học",
    "ĐTTX": "đào tạo từ xa",
    "CQ": "chính quy",
    "ĐHCQ": "đại học chính quy",
    "PĐT": "Phòng Đào tạo",
    "ĐRL": "điểm rèn luyện",
    "CNTT": "công nghệ thông tin",
    "NCKH": "nghiên cứu khoa học",
    "PFIEV": "chương trình Kỹ sư Việt-Pháp",
    "OISP": "Văn phòng Đào tạo Quốc tế",
    "TOEIC": "chuẩn ngoại ngữ chứng chỉ ngoại ngữ quốc tế",
    "IELTS": "chuẩn ngoại ngữ chứng chỉ ngoại ngữ quốc tế",
    "TOEFL": "chuẩn ngoại ngữ chứng chỉ ngoại ngữ quốc tế",
    "ĐHBK": "Đại học Bách khoa",
    "ĐHQG": "Đại học Quốc gia",
}

# Everyday phrasing -> the phrase used in the regulation. Matched case-insensitively.
SYNONYMS: dict[str, str] = {
    "đuổi học": "buộc thôi học",
    "bị đuổi": "buộc thôi học",
    "cảnh cáo học tập": "cảnh báo học tập",
    "cảnh cáo": "cảnh báo học tập",
    "bảo lưu": "nghỉ học tạm thời",
    "tạm dừng học": "buộc tạm dừng học",
    "rớt môn": "học phần không đạt điểm F",
    "trượt môn": "học phần không đạt điểm F",
    "rớt": "không đạt",
    "nợ môn": "tín chỉ nợ đọng học phần không đạt",
    "học lại": "học lại học phần không đạt",
    "thi lại": "học lại thi cải thiện điểm",
    "cải thiện": "học cải thiện điểm",
    "phúc khảo": "phúc tra bài thi",
    "chấm lại": "phúc tra bài thi",
    "ra trường": "tốt nghiệp",
    "bằng giỏi": "khen thưởng lúc tốt nghiệp xếp loại",
    "xếp loại bằng": "khen thưởng lúc tốt nghiệp xếp loại",
    "bằng hai": "song ngành bằng thứ hai",
    "bằng 2": "song ngành bằng thứ hai",
    "song bằng": "đào tạo song ngành",
    "ngành hai": "ngành thứ hai",
    "môn học": "học phần",
    "số môn": "số học phần",
    "đăng ký môn": "đăng ký học phần",
    "rút môn": "rút đăng ký học phần",
    "huỷ môn": "huỷ đăng ký học phần",
    "hủy môn": "huỷ đăng ký học phần",
    "vắng thi": "vắng mặt buổi thi điểm I",
    "hoãn thi": "hoãn thi điểm I",
    "điểm f": "loại không đạt điểm F",
    "điểm i": "điểm chưa hoàn thiện hoãn thi",
    "điểm w": "điểm rút đăng ký học phần",
    "điểm r": "điểm miễn học công nhận tín chỉ",
    "năm nhất": "trình độ năm thứ nhất",
    "năm hai": "trình độ năm thứ hai",
    "năm ba": "trình độ năm thứ ba",
    "năm tư": "trình độ năm thứ tư",
    "tiếng anh đầu ra": "chuẩn ngoại ngữ tốt nghiệp",
    "chuẩn đầu ra tiếng anh": "chuẩn ngoại ngữ tốt nghiệp",
    "anh văn đầu ra": "chuẩn ngoại ngữ tốt nghiệp",
    "học phí trễ": "lịch thanh toán học phí",
    "đóng học phí": "lịch thanh toán học phí",
    "hoàn tiền": "hoàn trả học phí",
    "chuyển ngành": "chuyển ngành chuyển chương trình",
    "chuyển trường": "chuyển trường chuyển nơi học",
    "tối đa bao nhiêu tín chỉ": "số tín chỉ đăng ký tối đa",
    "tối thiểu bao nhiêu tín chỉ": "số tín chỉ đăng ký tối thiểu",
    "học vượt": "đăng ký vượt số tín chỉ tối đa",
    "mã số sinh viên": "cấu trúc mã số người học",
    "mssv": "cấu trúc mã số người học",
    "lớp trưởng": "lớp sinh viên",
}


def strip_accents(text: str) -> str:
    """'Học phần' -> 'hoc phan'. Used to match queries typed without diacritics."""
    s = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def _compile(keys, flags=0) -> re.Pattern | None:
    """One alternation, longest key first so 'bằng hai' wins over 'bằng'."""
    keys = sorted((k for k in keys if k), key=len, reverse=True)
    if not keys:
        return None
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", flags)


_LONG_ACRONYMS = {k: v for k, v in ACRONYMS.items() if len(k) > 2}
_SHORT_ACRONYMS = {k: v for k, v in ACRONYMS.items() if len(k) <= 2}
_RE_LONG = _compile(_LONG_ACRONYMS, re.IGNORECASE)
_RE_SHORT = _compile(_SHORT_ACRONYMS)  # case-sensitive: only "TC", not "tc"
_RE_SYNONYM = _compile(SYNONYMS, re.IGNORECASE)
# Accent-free lookup, for queries typed as "hoc phan khong dat". Only keys long enough to be
# unambiguous go in: a three-letter accent-free form would match half the language.
_BARE: dict[str, str] = {}
for _table in (ACRONYMS, SYNONYMS):
    for _key, _value in _table.items():
        if len(_key) >= 5 or (" " in _key and len(_key) >= 4):
            _BARE.setdefault(strip_accents(_key), _value)
for _phrase in set(ACRONYMS.values()) | set(SYNONYMS.values()):
    _BARE.setdefault(strip_accents(_phrase), _phrase)
_RE_BARE = _compile(_BARE)


@dataclass
class Expansion:
    text: str  # query + the canonical phrases, what gets searched
    added: list[str] = field(default_factory=list)  # phrases that were appended
    original: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.added)


def has_diacritics(text: str) -> bool:
    return strip_accents(text) != text.lower()


def expand(query: str, enabled: bool = True) -> Expansion:
    """Append the regulation's wording for any abbreviation or slang found in the query."""
    if not enabled or not query.strip():
        return Expansion(query, [], query)
    found: list[str] = []

    def collect(table: dict[str, str], key: str) -> None:
        phrase = table.get(key) or table.get(key.upper()) or table.get(key.lower())
        if phrase and phrase.lower() not in query.lower() and phrase not in found:
            found.append(phrase)

    for pattern, table in ((_RE_LONG, _LONG_ACRONYMS), (_RE_SHORT, _SHORT_ACRONYMS),
                           (_RE_SYNONYM, SYNONYMS)):
        if pattern is None:
            continue
        for m in pattern.finditer(query):
            collect(table, m.group(1))

    # Query typed without diacritics: map what we can onto the real spelling.
    if not has_diacritics(query) and _RE_BARE is not None:
        for m in _RE_BARE.finditer(strip_accents(query)):
            phrase = _BARE[m.group(1)]
            if phrase not in found:
                found.append(phrase)

    text = f"{query} — {'; '.join(found)}" if found else query
    return Expansion(text, found, query)
