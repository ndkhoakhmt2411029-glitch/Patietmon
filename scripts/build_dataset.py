"""Bước 2.2: dựng bộ dữ liệu YOLO từ nhãn đã duyệt, chia tập theo video/nhóm."""
from patientmon import cli
from patientmon.dataset import build_yolo_dataset, load_meta
from patientmon.frames import load_manifest


def main():
    cfg, _ = cli.init(__doc__)
    d = cfg.dataset
    summary = build_yolo_dataset(
        load_manifest(cfg.paths.resolve("frames")), cfg.paths.resolve("frames"), cfg.paths.resolve("labels_reviewed"),
        cfg.paths.resolve("yolo_dataset"), d.names, d.ratios, d.seed, load_meta(cfg.paths.meta_path()),
    )
    for split, s in summary.items():
        print(f"{split}: {s['videos']} video, {s['frames']} khung, {s['boxes']} hộp")


if __name__ == "__main__":
    main()
