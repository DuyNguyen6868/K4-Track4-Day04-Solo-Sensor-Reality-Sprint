# Bước 5 · Failure case và cải tiến

Mỗi câu dưới đây được gắn một trong ba nhãn: **[Đo]** là kết quả tự đo, **[Nguồn]** là kết luận của paper, **[Giả thuyết]** là suy luận chưa kiểm chứng.

## Failure case đã chọn

Motion blur 15 px: camera đã mất một nửa khả năng phát hiện, nhưng hai tín hiệu sức khỏe đơn giản nhất đều không báo động đủ.

| Câu cần trả lời | Bằng chứng |
|---|---|
| Sensor gặp lỗi gì, ở mức nào? | Motion blur ngang, kernel 15 px, trên 128 ảnh COCO128. Ảnh minh họa: `000000000257.jpg` trong [fig3_before_after.png](../results/fig3_before_after.png) |
| Metric thay đổi ra sao? | [Đo] Recall 55.8% → 27.0%, tức còn 48.4% so với baseline. Precision 83.1% → 62.4%. Blur score trung vị 1285 → 138 |
| Thuật toán/tính năng bị ảnh hưởng thế nào? | [Đo] Recall object cỡ vừa 67.0% → 13.8%, cỡ nhỏ 25.3% → 0.7%, cỡ lớn 91.0% → 79.0%. [Giả thuyết] Với FCW/AEB, object cỡ vừa và nhỏ trong ảnh thường là người và xe ở xa, nên hệ thống mất khả năng phát hiện sớm trước khi mất khả năng phát hiện gần |
| Phương pháp còn hạn chế ở đâu? | Xem hai mục "Tín hiệu sức khỏe bỏ sót" và "Giới hạn" bên dưới |
| Nên làm gì tiếp? | Xem mục "Cải tiến đề xuất" |

## Tín hiệu sức khỏe bỏ sót ở đâu

**Confidence của detector không báo trước được.**

- [Đo] Từ baseline tới blur 15 px, confidence trung bình của các detection đúng chỉ đổi 0.68 → 0.65, trong khi recall giảm hơn một nửa.
- [Giả thuyết] Nguyên nhân là thiên lệch sống sót: confidence chỉ tính trên những object còn được phát hiện, tức những object lớn và rõ; object đã mất không góp vào trung bình.
- [Nguồn] N5 báo cáo confidence của YOLO tương quan 0.90 với mAP nhưng không cảnh báo trước khi detector hỏng, và mô tả nó là tín hiệu "reactive". N5 đo trên KITTI với mAP50:95, nên đây là kết luận cùng hướng chứ không phải cùng con số.

**Ngưỡng blur score tuyệt đối bỏ sót cảnh nhiều chi tiết.**

- [Đo] Ở blur 15 px, 29 trên 128 khung hình (22.7%) vẫn có blur score trên ngưỡng 310.3 và không bị gắn cờ. 29 khung hình này chứa 144 trên 344 object; số object phát hiện đúng trong đó giảm từ 64 xuống 20.
- [Đo] Blur score ở baseline trải từ 17 tới 29509 giữa các ảnh, rộng hơn nhiều so với mức thay đổi do nhòe gây ra trên một ảnh.
- [Đo] Ảnh `000000000257.jpg`: blur score 3080 ở baseline, 316 ở blur 25 px, không lần nào dưới ngưỡng, dù số object đúng giảm từ 13 xuống 1 trên 27.
- [Giả thuyết] Phương sai Laplacian đo lượng cạnh trong ảnh, vốn phụ thuộc nội dung cảnh. Một cảnh phố nhiều chi tiết bị nhòe vẫn có nhiều cạnh hơn một cảnh trời trống còn nét.

**Blur score đơn lẻ đọc sai lỗi quá sáng.**

- [Đo] Ở gain ×4, recall còn 55.7% so với baseline, nhưng blur score trung vị tăng từ 1285 lên 2858 và chỉ 3.1% khung hình bị gắn cờ nhòe. Cờ quá sáng bắt được 99.2%.
- [Đo] Chiều ngược lại, cờ quá sáng chỉ bật ở 5.5-9.4% khung hình khi lỗi là nhòe.
- [Đo] Cờ quá sáng lại quá nhạy: ở gain ×1.5 nó bật trên 79.7% khung hình trong khi recall chỉ giảm còn 97.4% so với baseline.

## Đặt cạnh nguồn

- [Nguồn] N5 định nghĩa detector "hỏng" khi mAP50:95 giảm 20% tương đối, và báo cáo motion blur có lead time ngắn nhất (0.10 đơn vị mức độ) trong các loại lỗi vì tác động đột ngột.
- [Đo] Nếu mượn ngưỡng 20% tương đối đó cho recall, benchmark này vượt ngưỡng ở khoảng giữa 5 px (còn 87.0%) và 9 px (còn 70.8%). Đây là áp ngưỡng của N5 lên một metric khác và dataset khác, không phải tái hiện số của N5.
- [Nguồn] N1 báo cáo Faster R-CNN ResNet-50 trên COCO còn 50.2% hiệu năng khi lấy trung bình 15 loại corruption. Con số này không so trực tiếp được với 48.4% ở trên: khác model, khác metric, khác tập ảnh, và là trung bình nhiều loại lỗi.

## Giới hạn của benchmark lớp học

- **Ảnh đã thấy khi huấn luyện:** COCO128 nằm trong tập train của YOLOv8n. Recall tuyệt đối không đại diện cho ảnh mới; chỉ nên đọc mức giảm tương đối.
- **Nhòe tổng hợp:** kernel ngang, đều trên toàn ảnh. Chưa có nhòe theo hướng bất kỳ, nhòe riêng vật thể chuyển động, hay rolling shutter.
- **Ngưỡng đặt và đánh giá trên cùng 128 ảnh:** tỉ lệ báo nhầm 10% là do cách đặt, chưa được kiểm tra trên ảnh khác.
- **Không phải dữ liệu lái xe:** COCO128 là ảnh đời thường; phân bố kích thước và khoảng cách object khác camera trước của xe.
- **Chưa đo:** mAP, hành vi trên video liên tiếp, latency đầu-cuối trên phần cứng nhúng, và kết hợp nhiều lỗi cùng lúc.

## Cải tiến đề xuất

**Thay ngưỡng blur score tuyệt đối bằng blur score chuẩn hóa theo cảnh, kết hợp với cờ quá sáng, và dùng điểm này để hạ trọng số camera thay vì dựa vào confidence.**

- **Cách chuẩn hóa:** chia blur score của khung hình hiện tại cho trung vị blur score của chính camera đó trong vài giây gần nhất. Cách này bám vào failure vừa thấy: sai lệch đến từ khác biệt giữa các cảnh, không phải từ bản thân phép đo.
- **Fallback khi bị gắn cờ:** hạ trọng số camera trong fusion, không coi "camera không thấy gì" là "đường trống", và dựa vào radar cho khoảng cách phía trước.
- **Cách kiểm chứng ở vòng sau:** chạy lại đúng benchmark này trên chuỗi video, giữ tỉ lệ báo nhầm ở baseline bằng 10%, rồi so tỉ lệ khung hình bị gắn cờ ở blur 5, 9, 15 px với các số hiện tại là 47.7%, 68.8%, 77.3%. Thêm một chỉ số: số object nằm trong các khung hình bị bỏ sót, hiện là 144 trên 344 ở 15 px.
