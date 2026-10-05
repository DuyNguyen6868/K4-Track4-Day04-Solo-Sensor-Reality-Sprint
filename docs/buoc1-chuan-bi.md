# Bước 1 · Chuẩn bị

Làm solo: một người đảm nhận cả bốn vai trò.

## Bảng chốt

| Nhóm cần chốt | Nội dung |
|---|---|
| Chủ đề | T1 — Camera degradation health score |
| Nền tảng, tính năng, sensor | Xe ADAS; tính năng phát hiện người và phương tiện phía trước (đầu vào cho cảnh báo va chạm FCW / phanh khẩn cấp AEB); camera RGB phía trước |
| Failure case | Motion blur: ảnh bị nhòe theo phương ngang khi thời gian phơi sáng dài (trời tối) kết hợp xe rung hoặc chuyển động nhanh |
| Claim ban đầu | Khi độ dài kernel nhòe tăng từ 0 lên 25 px, blur score (phương sai Laplacian) giảm và recall của detector giảm so với baseline; confidence trung bình cũng giảm |
| Metric và đơn vị | (1) Blur score = phương sai của ảnh sau lọc Laplacian, không đơn vị, càng thấp càng nhòe. (2) Recall = số object có nhãn được phát hiện đúng (IoU ≥ 0.5, đúng lớp) / tổng số object có nhãn, tính bằng %. (3) Confidence trung bình của các detection đúng, thang 0-1. (4) Latency suy luận mỗi ảnh, ms, để kiểm tra việc nhòe không làm đổi thời gian chạy. (5) Ba chỉ số đối chứng theo gợi ý của T1: exposure = độ sáng trung bình ảnh xám (0-255), saturation ratio = % điểm ảnh xám ≥ 250, và entropy histogram ảnh xám (bit); dự đoán gần như không đổi khi chỉ có nhòe, để cho thấy health score phân biệt được nhòe với lỗi phơi sáng |
| Baseline và điều kiện lỗi | Baseline: ảnh gốc (kernel 0 px). Điều kiện lỗi: cùng bộ ảnh đó, áp motion blur ngang với kernel 5, 9, 15, 25 px. Cùng model, cùng ngưỡng confidence, cùng cách tính metric cho mọi mức |
| Phân công | Solo — đọc nguồn, chạy thử, ghi số và trình bày đều do một người làm |

## Thiết lập thử nghiệm dự kiến

- **Dữ liệu:** COCO128 (128 ảnh có nhãn chuẩn), chỉ tính các lớp liên quan tới ADAS: person, bicycle, car, motorcycle, bus, truck, traffic light. Có nhãn chuẩn nên recall là metric thật, không phải proxy.
- **Model:** YOLOv8n pretrained, chạy CPU.
- **Yếu tố thay đổi duy nhất:** độ dài kernel motion blur (px).
- **Giới hạn đã biết:** blur tổng hợp bằng kernel tuyến tính đều, không mô phỏng rolling shutter hay nhòe phụ thuộc độ sâu; COCO không phải dữ liệu lái xe thuần; mẫu nhỏ nên kết quả chỉ cho thấy xu hướng.

## Tự kiểm tra

Người ngoài đọc bảng phải trả lời được hai câu:

- Thay đổi yếu tố nào? → Độ dài kernel motion blur, 0 → 25 px.
- Đo điều gì? → Blur score, recall, confidence trung bình và latency trên cùng bộ ảnh, cùng model.
