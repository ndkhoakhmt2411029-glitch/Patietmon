"""Giai đoạn 5 (bước 3): so sánh các checkpoint trên cùng một tập (mức cửa sổ, mức đoạn, tham số, tốc độ)."""
from patientmon import cli
from patientmon.actions.data import WINDOWS_FILE, load_windows
from patientmon.actions.evaluate import evaluate_checkpoint, format_confusion, format_table


def _args(parser):
    parser.add_argument("--checkpoints", nargs="+", required=True, help="các file best.pt cần so sánh")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--confusion", action="store_true", help="in thêm ma trận nhầm lẫn từng mô hình")


def main():
    cfg, args = cli.init(__doc__, _args)
    data = load_windows(cfg.paths.resolve("windows") / WINDOWS_FILE)
    results = [evaluate_checkpoint(c, data, args.split) for c in args.checkpoints]
    print(f"\n## Tập {args.split}\n{format_table(results)}")
    if args.confusion:
        for r in results:
            print("\n" + format_confusion(r))


if __name__ == "__main__":
    main()
