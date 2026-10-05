# K4-Track4-Day04-Solo-Sensor-Reality-Sprint

Lab Ngày 4 — Sensor Reality Sprint. Chủ đề **T1 · Camera degradation health score**: đo tác động của motion blur và quá sáng trên camera trước của xe ADAS lên khả năng phát hiện người và phương tiện, và kiểm tra tín hiệu sức khỏe nào báo trước được sự suy giảm đó.

## Kết quả chính

Tự đo với YOLOv8n trên 128 ảnh COCO128, 344 object thuộc 7 lớp ADAS:

| Điều kiện | Recall (%) | Confidence TB | Blur score | Khung hình bị gắn cờ nhòe (%) |
|---|---|---|---|---|
| Baseline | 55.8 | 0.68 | 1285 | 10.2 |
| Blur 9 px | 39.5 | 0.67 | 212 | 68.8 |
| Blur 15 px | 27.0 | 0.65 | 138 | 77.3 |
| Blur 25 px | 17.4 | 0.54 | 87 | 89.1 |

Ở blur 15 px, recall giảm hơn một nửa nhưng confidence gần như không đổi, và 29 trên 128 khung hình không bị ngưỡng blur score phát hiện.

![Quét motion blur](results/fig1_blur_sweep.png)

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| [reports/bao-cao-ca-nhan.md](reports/bao-cao-ca-nhan.md) | Báo cáo cá nhân, năm mục |
| [docs/buoc1-chuan-bi.md](docs/buoc1-chuan-bi.md) | Bài toán, claim, metric |
| [docs/buoc2-nguon-va-duong-chay.md](docs/buoc2-nguon-va-duong-chay.md) | Nguồn đã đọc và đường chạy |
| [docs/buoc3-4-benchmark-va-ket-qua.md](docs/buoc3-4-benchmark-va-ket-qua.md) | Thiết kế benchmark và bảng kết quả đầy đủ |
| [docs/buoc5-failure-case.md](docs/buoc5-failure-case.md) | Failure case, giới hạn, cải tiến |
| [src/benchmark.py](src/benchmark.py) | Chạy baseline và các điều kiện lỗi, ghi CSV và log |
| [src/plots.py](src/plots.py) | Vẽ ba hình từ CSV |
| [results/](results/) | `run.log`, `config.json`, bốn tệp CSV, ba hình PNG |
| [TEAMMATES.md](TEAMMATES.md) | Thành viên |

## Chạy lại

Cần Python 3.12, chạy trên CPU, mất khoảng hai phút. Lệnh cho PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ultralytics opencv-python matplotlib pandas
.\.venv\Scripts\python.exe src\benchmark.py --kernels 5 9 15 25 --gains 1.5 2.5 4.0
.\.venv\Scripts\python.exe src\plots.py
```

Script tự tải COCO128 (khoảng 7 MB) vào `datasets/` và trọng số `yolov8n.pt`; cả hai không nằm trong repo. Phiên bản thư viện đã dùng được ghi trong [results/config.json](results/config.json).

## Nguồn

- Aher, "Safety-Critical Camera Reliability Monitoring for ADAS via Degradation-Aware Uncertainty Pattern Analysis", 2026 — https://arxiv.org/abs/2605.05439
- Michaelis và cộng sự, "Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming", 2019 — https://arxiv.org/abs/1907.07484
- `imagecorruptions` — https://github.com/bethgelab/imagecorruptions
- COCO128 — https://docs.ultralytics.com/datasets/detect/coco128/
