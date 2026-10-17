"""Bước 2.3: fine-tune YOLOv8 trên bộ dữ liệu đã dựng (nên chạy trên Colab/Kaggle có GPU)."""
from patientmon import cli
from patientmon.detection.train import train_detector


def main():
    cfg, _ = cli.init(__doc__)
    best = train_detector(cfg, cfg.paths.resolve("yolo_dataset") / "data.yaml")
    print(f"Trọng số tốt nhất: {best}")


if __name__ == "__main__":
    main()
