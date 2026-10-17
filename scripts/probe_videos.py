"""Bước 1.2: kiểm tra video (đọc được không, FPS, độ phân giải, thời lượng) và đếm thư mục ảnh."""
from patientmon import cli
from patientmon.sources import ImageFolderSource, discover_sources
from patientmon.video import find_videos, probe, probe_report, write_probe_csv


def main():
    cfg, _ = cli.init(__doc__)
    root = cfg.paths.resolve("videos")
    infos = [probe(p) for p in find_videos(root)]
    print(probe_report(infos) if infos else "Không có file video.")
    if infos:
        out = root.parent / "video_report.csv"
        write_probe_csv(infos, root, out)
        print(f"Đã ghi {out}")
    folders = [s for s in discover_sources(root, cfg.source.image_kind, cfg.source.image_fps) if isinstance(s, ImageFolderSource)]
    if folders:
        print(f"Thư mục ảnh ({cfg.source.image_kind}): {len(folders)}, tổng {sum(s.n_frames for s in folders)} ảnh")


if __name__ == "__main__":
    main()
