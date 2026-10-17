"""Giai đoạn 5 (bước 1): gán nhãn từ tên thư mục/file, cắt cửa sổ skeleton, chuẩn hóa, chia tập theo người."""
from patientmon import cli
from patientmon.actions.data import build_window_dataset


def main():
    cfg, _ = cli.init(__doc__)
    summary = build_window_dataset(cfg.paths.resolve("skeletons"), cfg.paths.resolve("windows"), cfg.action)
    for split, s in summary.items():
        print(f"{split}: {s['windows']} cửa sổ, {s['groups']} nhóm, theo lớp {s['per_class']}")


if __name__ == "__main__":
    main()
