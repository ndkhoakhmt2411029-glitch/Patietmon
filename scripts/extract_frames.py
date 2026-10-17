"""Bước 1.3: cắt khung hình để gán nhãn khung bao (từ video hoặc thư mục ảnh; chạy lại được)."""
from patientmon import cli
from patientmon.frames import extract_all


def main():
    cfg, _ = cli.init(__doc__)
    records = extract_all(cfg.paths.resolve("videos"), cfg.paths.resolve("frames"), cfg.frames, cfg.source)
    print(f"Tổng số khung: {len(records)}")


if __name__ == "__main__":
    main()
