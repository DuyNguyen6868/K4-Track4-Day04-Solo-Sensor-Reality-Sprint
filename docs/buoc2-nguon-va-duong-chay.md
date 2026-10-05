# Bước 2 · Nguồn và đường chạy

Chủ đề T1 — Camera degradation health score. Failure case: motion blur trên camera trước của xe ADAS.

## Nguồn đã đọc

| # | Nguồn | Link | Đã đọc phần nào |
|---|---|---|---|
| N5 | **Nguồn chính.** Aher, "Safety-Critical Camera Reliability Monitoring for ADAS via Degradation-Aware Uncertainty Pattern Analysis", arXiv:2605.05439 (06/05/2026) | https://arxiv.org/abs/2605.05439 | Abstract, mục III-A (công thức GSHI), mục VI-A đến VI-C, Bảng III-V, mục VII (limitations) |
| N6 | Que và Yao, "A Degradation-Tolerance Benchmark for Camera-Only End-to-End Driving" (DriveDegrade), arXiv:2608.29005 (29/08/2026) | https://arxiv.org/abs/2608.29005 | Chỉ abstract |
| N7 | Xie và cộng sự, "Benchmarking and Improving Bird's Eye View Perception Robustness in Autonomous Driving" (RoboBEV), TPAMI 2025, arXiv:2405.17426 | https://arxiv.org/abs/2405.17426 · https://github.com/Daniel-xsy/RoboBEV | Chỉ abstract |
| N1 | Michaelis và cộng sự, "Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming", arXiv:1907.07484 (bản sửa 31/03/2020), NeurIPS 2019 ML4AD | https://arxiv.org/abs/1907.07484 | Abstract, mục 1.1-3.3, Bảng 1-2, Hình 3 và 5 |
| N2 | Repo benchmark của N1 | https://github.com/bethgelab/robust-detection-benchmark | README |
| N3 | Thư viện `imagecorruptions` 1.1.2 (03/12/2020), mã tạo corruption của N1 | https://github.com/bethgelab/imagecorruptions | README, hàm `motion_blur` trong `imagecorruptions/corruptions.py` (nhánh master) |
| N4 | Tài liệu dataset COCO128 của Ultralytics | https://docs.ultralytics.com/datasets/detect/coco128/ | Trang mô tả dataset |

Các nguồn S1-S9 trong PDF chưa dùng: S5 và S6 gần chủ đề nhất nhưng là benchmark 3D / đa cảm biến, không phải camera 2D với motion blur.

## Nguồn chính N5 (2026) theo câu hỏi

| Câu hỏi | Ghi chép |
|---|---|
| Phương pháp nhận gì và tạo gì? | Một ảnh RGB → loại suy giảm (12 loại), mức độ từng loại, điểm sức khỏe GSHI trong [0, 1] và bản đồ bất định theo vùng. Mạng EfficientNet-B2 đa nhiệm, không cần phản hồi từ tác vụ phía sau |
| Điểm sức khỏe tính thế nào? | GSHI = tích của (1 − mức độ loại i) mũ (trọng số rủi ro), nên một lỗi nặng đơn lẻ như motion blur hay che ống kính đủ kéo điểm xuống. Trọng số lớn nhất dành cho che ống kính, glare, motion blur, defocus |
| Nguồn đo chất lượng bằng gì? | MAE của điểm sức khỏe; tương quan Pearson giữa GSHI và mAP50:95 của YOLOv8; lead time = khoảng cách mức độ giữa lúc cảnh báo (GSHI < 0.8) và lúc detector "hỏng" (mAP50:95 giảm 20% tương đối). Dữ liệu: KITTI với suy giảm tổng hợp, quét mức độ 0.0-1.0; DAWN để thử zero-shot trên thời tiết thật. Detector: YOLOv8n pretrained COCO, đóng băng |
| Chạy được ở lớp không? | Không. Trang arXiv không nêu link code hay trọng số; cần huấn luyện mạng giám sát trên KITTI |
| Sẽ tái hiện phần nào? | Phần baseline của N5: dùng chỉ số chất lượng ảnh không cần nhãn và confidence của YOLO làm tín hiệu sức khỏe, rồi so với chất lượng detector khi mức suy giảm tăng. Cùng detector YOLOv8n đóng băng như N5 |

Kết luận của N5 (không phải số tự đo):

- GSHI giảm đơn điệu theo mức độ, MAE 0.064, cảnh báo sớm trung bình 0.47 ± 0.25 đơn vị mức độ trước khi YOLOv8 hỏng (abstract, Bảng III).
- Motion blur có lead time ngắn nhất, 0.10, vì tác động đột ngột lên cấu trúc ảnh; tương quan với mAP là 0.98 (Bảng III, mục VI-B).
- Confidence của YOLO tương quan 0.90 với mAP nhưng không cảnh báo sớm; các chỉ số IQA BRISQUE, NIQE, PIQE tương quan 0.59, 0.73, 0.72 và cũng không cảnh báo sớm (Bảng V).

Giới hạn do N5 nêu (mục VII): suy giảm tổng hợp không tái tạo hết hỗn hợp lỗi ngoài thực tế; chưa hiệu chuẩn định lượng trên dữ liệu thật; ngưỡng phải hiệu chuẩn lại theo từng xe, vị trí camera và miền vận hành; chưa có rolling shutter, gạt nước, điểm ảnh chết; chỉ xử lý từng khung hình đơn; đánh giá có kiểm soát mới giới hạn ở KITTI.

Hai nguồn mới khác, chỉ đọc abstract: N6 báo cáo blur, JPEG và raindrop làm hỏng planning nhiều nhất trong 16 loại corruption; N7 đánh giá 33 mô hình BEV dưới các corruption camera ở ba mức, trong đó có motion blur.

## Nguồn nền N1-N4 theo câu hỏi

| Câu hỏi | Ghi chép |
|---|---|
| Phương pháp nhận gì và tạo gì? | N1: ảnh test sạch + detector đã huấn luyện → ảnh bị corruption (15 loại × 5 mức) → điểm AP trên từng loại và từng mức. N3: một ảnh kích thước bất kỳ + tên corruption + mức 1-5 → ảnh đã bị làm hỏng |
| Nguồn đo chất lượng bằng gì? | P = AP trên dữ liệu sạch (AP50 với PASCAL VOC; AP trung bình IoU 50-95% với COCO và Cityscapes). mPC = trung bình P trên 15 corruption × 5 mức. rPC = mPC / P sạch. Dữ liệu: PASCAL-C (VOC2007 test), COCO-C (COCO 2017 val), Cityscapes-C (Cityscapes val). Model: họ Faster/Mask/Cascade R-CNN, RetinaNet, HTC chạy bằng mmdetection |
| Chạy được ở lớp không? | Đường chạy gốc: không. Cần mmdetection, COCO 2017 val và 75 biến thể corruption cho mỗi ảnh; máy đang dùng chỉ có CPU. Riêng N3 cài được bằng `pip3 install imagecorruptions`, nhưng bản phát hành cuối là năm 2020 và import `pkg_resources`, `numba`, `scipy.ndimage.interpolation` nên có rủi ro không tương thích Python 3.12 |
| Sẽ tái hiện phần nào? | Benchmark mô phỏng nhỏ theo đúng ý tưởng của N1: cùng ảnh, cùng model, chỉ đổi mức corruption, rồi so với kết quả sạch. Chỉ làm hai loại corruption, mỗi lần đổi một loại (motion blur là chính, quá sáng là phụ). Recall là metric thật vì có nhãn chuẩn; blur score, saturation ratio, exposure, entropy là chỉ số sức khỏe ảnh không cần nhãn |
| Cần trích lại gì khi báo cáo? | Link N1-N4, phiên bản thư viện (bảng dưới), dataset COCO128, lệnh chạy |

## Kết luận của nguồn (không phải số tự đo)

- Các detector chuẩn mất hiệu năng mạnh trên ảnh hỏng, còn khoảng 30-60% hiệu năng gốc (N1, abstract).
- Faster R-CNN ResNet-50 trên COCO: P sạch 36.3 AP, mPC 18.2 AP, rPC 50.2% — trung bình trên cả 15 loại corruption, không riêng motion blur (N1, Bảng 1).
- Với các backbone khác nhau, mọi corruption trừ nhóm blur gây mức phạt gần như cố định, không phụ thuộc hiệu năng sạch (N1, mục 3.2).
- Motion blur trong N3 dùng kernel Gaussian 1 chiều, 5 mức (radius, sigma) = (10, 3), (15, 5), (15, 8), (15, 12), (20, 15), góc ngẫu nhiên trong ±45°.

Giới hạn do chính nguồn nêu:

- Các corruption dùng để đo độ bền với nhiễu chưa từng thấy, không dùng làm data augmentation khi huấn luyện (N1 mục 2.1, N3 README).
- COCO128 là 128 ảnh đầu của COCO train2017, dùng chung cho train và val, mục đích là kiểm tra nhanh pipeline (N4).

## Đường chạy tối thiểu đã chốt

| Mục | Lựa chọn |
|---|---|
| Kiểu chạy | Benchmark mô phỏng degradation, script tự viết |
| Model | YOLOv8n pretrained trên COCO, chạy CPU |
| Dữ liệu | COCO128, lọc 7 lớp ADAS: person, bicycle, car, motorcycle, bus, truck, traffic light |
| Degradation chính | Motion blur ngang, kernel đều dài 0 (baseline), 5, 9, 15, 25 px, viết bằng OpenCV |
| Degradation phụ | Quá sáng: nhân cường độ với 1.5, 2.5, 4.0 rồi cắt ở 255. Dùng để kiểm tra health score có phân biệt được hai loại lỗi |
| Metric | Recall (%), precision (%), confidence trung bình, blur score, saturation ratio (%), exposure, entropy (bit), latency (ms) |

Khác biệt so với nguồn và lý do:

- **Không chạy mmdetection + COCO-C:** quá nặng cho CPU trong 120 phút.
- **Tự viết motion blur thay vì gọi N3:** N3 chọn góc ngẫu nhiên mỗi ảnh và có rủi ro cài đặt trên Python 3.12. Kernel ngang cố định cho kết quả lặp lại được và độ dài tính bằng px dễ diễn giải. Hệ quả: các mức của mình không tương ứng với severity 1-5 của N1, nên không so trực tiếp với số của N1.
- **Recall thay vì AP:** recall ở một ngưỡng confidence cố định dễ giải thích cho tính năng an toàn (bỏ sót người/xe) và tính nhanh. Nó không phản ánh false positive.
- **YOLOv8n thay vì họ R-CNN:** nhẹ, chạy được CPU. N1 không đo model này.

Điều benchmark này chưa chứng minh được:

- YOLOv8n được huấn luyện trên COCO train2017, tức đã thấy 128 ảnh này. Recall tuyệt đối ở baseline sẽ cao hơn thực tế; chỉ nên đọc mức sụt giảm tương đối giữa các mức nhòe.
- Blur tổng hợp, đồng đều toàn ảnh; chưa có rolling shutter, nhòe theo độ sâu hay nhòe riêng từng vật thể chuyển động.
- 128 ảnh, không phải cảnh lái xe thuần; kết quả chỉ cho thấy xu hướng.

## Khả năng tái hiện

Phiên bản thực tế trong môi trường:

| Thành phần | Phiên bản |
|---|---|
| Python | 3.12.10 |
| torch (CPU) | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| ultralytics | 8.4.173 |
| opencv-python | 5.0.0.93 |
| numpy | 2.5.2 |
| matplotlib / pandas | 3.11.2 / 3.0.6 |
| Trọng số | `yolov8n.pt` từ ultralytics/assets release v8.4.0 |

Đã chạy thử: YOLOv8n suy luận ảnh mẫu `bus.jpg` của Ultralytics trên CPU mất khoảng 90 ms (một lần đo, sau một lần khởi động), phát hiện 1 bus và 4 person. Ước tính 128 ảnh × 5 mức nhòe chạy dưới vài phút.

Lệnh setup (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ultralytics opencv-python matplotlib pandas
```

Lệnh đã thực sự dùng ở Bước 4:

```powershell
.\.venv\Scripts\python.exe src\benchmark.py --kernels 5 9 15 25 --gains 1.5 2.5 4.0
.\.venv\Scripts\python.exe src\plots.py
```

Script tự tải COCO128 (khoảng 7 MB) vào `datasets/`; `yolov8n.pt` được Ultralytics tải ở lần chạy đầu. Baseline luôn được chạy, không cần truyền kernel 0.

## Tự kiểm tra

- **Nhận gì, trả gì, đo bằng gì:** nhận ảnh camera và detector có sẵn; trả về ảnh đã nhòe ở từng mức cùng kết quả phát hiện; đo bằng recall so với nhãn chuẩn và các chỉ số sức khỏe ảnh.
- **Phép thử sẽ chạy:** cùng 128 ảnh, cùng YOLOv8n, baseline cộng bốn mức nhòe và ba mức quá sáng, một lệnh `src\benchmark.py`.
