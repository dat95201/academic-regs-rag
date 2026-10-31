# Báo cáo đồ án — mã nguồn LaTeX

Báo cáo môn CO5179: *Xây dựng hệ thống RAG hỗ trợ truy vấn quy định học vụ*.

## Biên dịch

```bash
./build.sh              # tự dò: dùng xelatex nếu có, không thì dùng Docker
./build.sh --docker     # ép dùng Docker (không cần cài LaTeX)
./build.sh --clean      # xoá file trung gian
```

Kết quả: `BaoCao-RAG-QuyDinhHocVu-CO5179.pdf`.

**Bắt buộc dùng XeLaTeX**, không dùng pdfLaTeX: báo cáo viết tiếng Việt Unicode và gọi font
theo tên hệ thống. Trên macOS cài bằng `brew install --cask mactex-no-gui`.

## Bố cục

```
main.tex            trang bìa, mục lục, gọi các chương
preamble.tex        font, màu, kiểu tiêu đề, bảng, khung ghi chú, đầu/chân trang
sections/           mỗi chương một file, đánh số theo thứ tự xuất hiện
img/                hình minh hoạ (diagram + ảnh chụp màn hình)
```

## Font

| Vai trò | Font | Vì sao |
|---|---|---|
| Thân bài | TeX Gyre Pagella | Bản TeX Gyre của Palatino, phủ đủ tiếng Việt, dễ đọc khi in dài |
| Tiêu đề, bảng, chú thích | TeX Gyre Heros | Bản TeX Gyre của Helvetica; nét sạch, tương phản rõ với Pagella |
| Mã nguồn | Latin Modern Mono | Phủ đủ tiếng Việt dựng sẵn |

Cả ba đều nằm trong TeX Live nên có sẵn ở mọi bản cài, kể cả image Docker dùng để build.

**Đừng đổi font mã nguồn sang DejaVu Sans Mono.** Font đó thiếu 21/49 ký tự tiếng Việt dựng
sẵn (`ế ồ ẩ ả ấ ầ ắ ẵ ẻ ề ể ễ` …). Khi thiếu glyph, HarfBuzz tách ký tự thành nền + dấu rời,
mà trong font đơn cách dấu rời chiếm hẳn một ô, nên chữ vỡ: `nếu` hiện thành `nêú`. Nếu cần
đổi, kiểm tra font mới có đủ dải U+1EA0–U+1EF9 trước.

## Quy ước khi sửa nội dung

- Dùng `\code{...}` cho tên biến, tên hàm, đường dẫn. Nhớ thoát dấu gạch dưới: `\code{top\_k\_final}`.
- Khối mã dùng môi trường `codeblock` (không cần thoát ký tự bên trong).
- Khung ghi chú dùng `\begin{notebox}[title={...}]`.
- Tham chiếu chéo bằng `\label` / `\ref`, đừng viết số hình và số mục bằng tay — chạy
  `./build.sh` hai lượt là số tự khớp.
