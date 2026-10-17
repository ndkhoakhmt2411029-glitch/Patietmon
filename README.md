# patientmon: giai đoạn 0 đến 5

Phát hiện hành động bất thường của bệnh nhân từ video. Phần này xây tới **Giai đoạn 5** theo đúng 8 giai đoạn của đồ án:
dữ liệu (1) → phát hiện người YOLOv8 (2) → ByteTrack (3) → pose (4) → phân loại hành động ST-GCN++ và các mô hình so sánh (5).
Giai đoạn 6 (FallNet), 7 (quyết định cảnh báo), 8 (ONNX/TensorRT) cắm thêm vào mà không phải sửa phần đã có.

## Đầu vào được nhận

| Loại | Cách đặt | Ghi chú |
|---|---|---|
| File video | `.mp4 .avi .mov .mkv .webm .m4v` trong `<data_root>/videos` (cho phép thư mục con) | |
| Thư mục ảnh chụp liên tiếp (như UP-Fall) | Thư mục nào có ảnh (`.jpg .jpeg .png .bmp`) thì là một nguồn | **Bắt buộc đặt `source.image_fps`**; ảnh sắp theo tên (số trong tên được so như số) |
| Ảnh đơn lẻ, không liên quan thời gian | Đặt `source.image_kind: stills` | Chỉ dùng cho Giai đoạn 2 (huấn luyện YOLO) |

ID của mỗi nguồn là đường dẫn tương đối viết không dấu, nối bằng `__` (ví dụ `S01__fall__Camera1`); ID này dùng để suy ra nhãn và nhóm.

## Quy trình

```bash
pip install -e .                          # Colab/Kaggle: trong thư mục dự án
python scripts/probe_videos.py            # kiểm tra video, đếm thư mục ảnh

# --- Giai đoạn 2, nhánh fine-tune YOLO (tùy chọn, cần nhãn khung bao) ---
python scripts/extract_frames.py          # cắt khung để gán nhãn
python scripts/prelabel.py                # nhãn nháp; duyệt/sửa rồi đặt vào labels_reviewed
python scripts/build_dataset.py           # chia tập theo video/người
python scripts/train_detector.py          # xong thì đặt detector.weights = best.pt, detector.class_map = null
python scripts/evaluate_detector.py --finetuned <best.pt> --split val --by condition

# --- Giai đoạn 2-4: trích skeleton (chạy lại được, bỏ qua nguồn đã xong) ---
python scripts/extract_skeletons.py

# --- Giai đoạn 5 ---
python scripts/build_windows.py           # nhãn từ tên thư mục, cửa sổ, chuẩn hóa, chia tập theo người
python scripts/train_action.py --model stgcnpp     # lặp lại với lstm, gru, tcn
python scripts/evaluate_action.py --checkpoints runs/action/stgcnpp/best.pt runs/action/lstm/best.pt --split test --confusion
```

Trên Colab: gắn Drive và đặt `paths.data_root` thành đường dẫn tuyệt đối trên Drive. Huấn luyện nên dùng GPU.

## Những chỗ bạn phải điền trong `configs/config.yaml`

1. **`source.image_fps`** nếu có chuỗi ảnh liên tiếp (số ảnh mỗi giây). Sai FPS làm tốc độ chuyển động sai và ST-GCN++ học lệch.
2. **`action.label_rules`**: cách suy ra nhãn từ ID nguồn. Luật đầu tiên khớp thắng, không phân biệt hoa thường, ví dụ
   `- {pattern: "walk", label: walk}`. Để trống thì `build_windows.py` báo lỗi (cố ý, không đoán nhãn thay bạn).
3. **`action.group_pattern`**: regex bắt mã người (ví dụ `"^s(\\d+)"`), để mọi đoạn của cùng một người luôn nằm cùng một tập.
4. **`skeleton.target_fps`**: đặt chung (ví dụ 15) nếu các nguồn có FPS khác nhau; `build_windows.py` cảnh báo khi FPS lệch nhau.

## Thiết kế (và lý do)

- **Giao diện thay được:** `Detector`, `Tracker`, `PoseEstimator`, mô hình phân loại. Đổi YOLOv8 hay thêm mô hình chỉ cần cài đặt lại một lớp.
- **Pose top-down:** YOLOv8 tìm người, ByteTrack gán ID, rồi mô hình pose chạy riêng trên từng khung người đã nới lề (`YoloPoseEstimator`, bố cục COCO-17).
- **Skeleton lưu ra file** (`skeletons/<nguồn>.npz`, dạng bảng dài: khung, người, 17 keypoint, hộp), nên huấn luyện không phải chạy lại YOLO.
- **Chuẩn hóa một hàm duy nhất** (`skeleton/normalize.py`) cho cả huấn luyện lẫn chạy thật: nội suy keypoint tin cậy thấp, tịnh tiến về hông trung bình của CẢ cửa sổ (giữ chuyển động đi xuống khi ngã), chia cho độ dài thân.
- **Chia tập theo người/nhóm**, không theo cửa sổ, để tránh rò rỉ giữa train và test.
- **Mọi mô hình dùng chung vòng huấn luyện và dữ liệu**, nên bảng so sánh (ST-GCN++, LSTM, GRU, TCN) công bằng; `evaluate_action.py` báo macro-F1, recall từng lớp, accuracy mức cửa sổ và mức đoạn, số tham số, ms/cửa sổ.

## Giới hạn cần biết

- **Nhãn một-cho-cả-đoạn còn nhiễu:** đoạn "ngã" gồm cả lúc đứng trước khi ngã và nằm sau khi ngã. Coi kết quả là phép đo thô; mốc thời gian từng hành động (Giai đoạn sau) sẽ chính xác hơn.
- **`max_tracks_per_clip: 1`** hợp với đoạn chỉ có một người. Với phòng bệnh nhiều người cần quy tắc chọn đúng bệnh nhân (Giai đoạn 7).
- **ST-GCN++ là cài đặt lại bằng PyTorch thuần theo mô tả của PYSKL**, có thể không giống hệt bản gốc; hãy ghi rõ trong báo cáo.
- **ByteTrack tự cài đặt** (numpy + scipy). Khác bản gốc ở bước ghép (dùng `linear_sum_assignment` rồi loại cặp quá ngưỡng thay cho lapjv) và không có tùy chọn `fuse_score`. Giống bản gốc: track mới chỉ được xuất từ khung thứ hai.
- Kết quả trên dữ liệu quay mô phỏng chưa chứng minh chạy tốt ở bệnh viện thật.

## Kiểm thử

```bash
python -m unittest discover -s tests -v     # hoặc pytest
```

Đã chạy và qua (62 kiểm thử): nguồn video/ảnh, cắt khung, ByteTrack, chuẩn hóa skeleton, cửa sổ, nhãn, chia tập, chỉ số, đồ thị, cấu hình; kèm kiểm tra bằng cách cố ý phá logic để xác nhận kiểm thử bắt được lỗi, và chạy thử các script dòng lệnh trên dữ liệu tổng hợp.

**Chưa chạy được ở nơi viết code (không có mạng, không có PyTorch/Ultralytics):**
- 7 kiểm thử PyTorch (`tests/test_models.py`: hình dạng đầu ra, lan truyền ngược, các luồng đặc trưng, tăng cường dữ liệu, huấn luyện và đánh giá thử trên dữ liệu tổng hợp) tự bỏ qua khi thiếu torch. **Hãy chạy chúng trên Colab trước tiên.**
- Phần gọi Ultralytics: `detection/yolo.py`, `detection/train.py`, `pose/yolo_pose.py` và các script `prelabel`, `train_detector`, `evaluate_detector`, `extract_skeletons`.

Nếu có lỗi, gửi nguyên nội dung báo lỗi; phiên bản thư viện được ghi vào `run_info.json` mỗi lần huấn luyện.
