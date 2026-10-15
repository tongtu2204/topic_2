# Slide Chuyên đề 2 – Phân tích và dự báo tỷ giá USD/VND (47 trang)

## Bản cập nhật

Bản này sửa **trực tiếp từ bộ Beamer 43 trang** đã làm trước đó. Toàn bộ nội dung của Phần VI (Mô hình Markov Switching) gồm 4 slide cũ được **thay bằng 8 slide biên tập lại** từ `slides_ms_ar_usdvnd_2026.pdf` mà người dùng cung cấp. Các công thức, bảng và phần chữ là mã LaTeX chỉnh sửa được. Hình P5.1 và P5.3 được lấy từ báo cáo gốc và giữ độ phân giải tốt.

- Trang 1: Bìa.
- Trang 2: Mục lục bảy phần.
- Trang 3–33: Phần I–V, không thay đổi nội dung.
- Trang 34: Trang phân cách **Phần VI – Mô hình Markov Switching**.
- **Trang 35–42: 8 slide Markov mới**, lần lượt gồm: mục tiêu/phạm vi; hai trạng thái; cách đọc ma trận; đặc tả và dự báo; tham số ước lượng; phân tích hai chế độ; test ngoài mẫu 2026; kết luận và giới hạn.
- Trang 43: Trang phân cách Phần VII.
- Trang 44–46: So sánh và kết luận tổng hợp, giữ lại từ bản cũ.
- Trang 47: Trang cảm ơn.

**Tổng 47 trang = 43 trang cũ − 4 slide Markov cũ + 8 slide Markov mới.**

## Những nguyên tắc được giữ lại

- Tông đỏ `mainred`, header/breadcrumb, thanh tiêu đề và footer phân đoạn của bộ Beamer hiện hành.
- Ghi chú đối chiếu ở góc phải dưới không có nhãn "Nguồn".
- Phần VII vẫn sử dụng kết quả tổng hợp trên 175 phiên test, phân biệt kết quả số học với ý nghĩa thống kê. Không tuyên bố mô hình Markov vượt Random Walk có ý nghĩa thống kê khi chưa có kiểm định tổn thất ghép cặp.
- Các tham số và KPI Markov lấy từ file Markov mới và đối chiếu với báo cáo Phần II, mục 7.5. Không tái ước lượng mô hình.

## Các tệp

- `main.tex`: mã nguồn chính, bao gồm toàn bộ 47 slide.
- `assets/`: 21 hình biểu đồ gốc dùng trong báo cáo.
- `README.md`: hướng dẫn và mô tả các thay đổi.

## Biên dịch trên Overleaf

1. Upload tệp ZIP dưới dạng dự án Overleaf.
2. Đặt `main.tex` làm main document.
3. Menu → Compiler → **XeLaTeX**.
4. Recompile (nên chạy hai lần để hiện đúng tổng số slide ở footer).

Phần VI nằm giữa `\section{VI. Mô hình Markov Switching}` và `\section{VII. So sánh và kết luận}` trong `main.tex`.
