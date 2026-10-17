"""Bước 2.1: tạo nhãn nháp bằng mô hình có sẵn và hàng đợi duyệt (khung trống/không chắc xếp trước).
Nhãn nháp CHƯA dùng để huấn luyện; hãy duyệt/sửa rồi đặt vào thư mục labels_reviewed."""
from patientmon import cli
from patientmon.detection import YoloDetector
from patientmon.frames import load_manifest
from patientmon.labels import prelabel


def main():
    cfg, _ = cli.init(__doc__)
    p = cfg.prelabel
    detector = YoloDetector(p.weights, p.conf, p.iou, p.imgsz, p.device, p.class_map)
    frames_dir = cfg.paths.resolve("frames")
    queue = prelabel(detector, load_manifest(frames_dir), frames_dir, cfg.paths.resolve("labels_auto"), p.batch, p.confident_score)
    print(f"Hàng đợi duyệt: {queue}")


if __name__ == "__main__":
    main()
