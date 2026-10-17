"""Bước 2.4: so sánh mô hình có sẵn với mô hình đã fine-tune trên cùng một tập; có thể tách theo cột siêu dữ liệu."""
from patientmon import cli
from patientmon.dataset import load_index, load_meta
from patientmon.detection import YoloDetector
from patientmon.evaluation import evaluate_grouped, format_table


def _args(parser):
    parser.add_argument("--finetuned", help="đường dẫn best.pt của mô hình đã fine-tune")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--by", help="tên cột trong meta_file để tách kết quả (ví dụ condition)")


def main():
    cfg, args = cli.init(__doc__, _args)
    e = cfg.evaluate
    out_dir = cfg.paths.resolve("yolo_dataset")
    rows = load_index(out_dir, args.split)
    meta = load_meta(cfg.paths.meta_path())
    key = (lambda r: meta.get(r["video_id"], {}).get(args.by) or "(không rõ)") if args.by else (lambda r: "tất cả")
    detectors = {"có sẵn": YoloDetector(e.baseline_weights, e.conf, 0.7, e.imgsz, None, e.baseline_class_map)}
    if args.finetuned:
        detectors["fine-tune"] = YoloDetector(args.finetuned, e.conf, 0.7, e.imgsz, None, None)
    for name, detector in detectors.items():
        results = evaluate_grouped(detector, rows, out_dir, key, target_class=e.target_class, iou_thr=e.iou_match,
                                   report_conf=e.report_conf, batch_size=e.batch)
        print(f"\n## Mô hình {name} - tập {args.split}\n{format_table(results)}")


if __name__ == "__main__":
    main()
