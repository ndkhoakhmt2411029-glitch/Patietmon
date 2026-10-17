"""Giai đoạn 2-4: phát hiện người (YOLOv8) -> theo dõi (ByteTrack) -> pose riêng trên từng khung người -> lưu skeleton.
Mỗi nguồn (video hoặc thư mục ảnh) một file .npz; nguồn đã xong được bỏ qua nên chạy lại được sau khi Colab ngắt."""
import logging

from patientmon import cli
from patientmon.detection import YoloDetector
from patientmon.pose import YoloPoseEstimator
from patientmon.skeleton import extract_skeletons
from patientmon.sources import discover_sources
from patientmon.tracking import ByteTracker

log = logging.getLogger("extract_skeletons")


def main():
    cfg, _ = cli.init(__doc__)
    d, tr, p = cfg.detector, cfg.tracking, cfg.pose
    sources = discover_sources(cfg.paths.resolve("videos"), cfg.source.image_kind, cfg.source.image_fps)
    detector = YoloDetector(d.weights, d.conf, d.iou, d.imgsz, d.device, d.class_map)
    pose = YoloPoseEstimator(p.weights, p.imgsz, p.margin, p.device)

    def tracker_factory(fps):
        return ByteTracker(tr.high_thresh, tr.low_thresh, tr.new_thresh, tr.buffer, tr.match_thresh, frame_rate=fps)

    out = cfg.paths.resolve("skeletons")
    for n, source in enumerate(sources, 1):
        path = out / f"{source.id}.npz"
        if path.exists():
            log.info("[%d/%d] %s: đã có, bỏ qua", n, len(sources), source.id)
            continue
        extract_skeletons(source, detector, tracker_factory, pose, cfg.skeleton.target_fps, d.batch).save(path)
        log.info("[%d/%d] %s: đã lưu %s", n, len(sources), source.id, path)


if __name__ == "__main__":
    main()
