"""Giai đoạn 5 (bước 2): huấn luyện bộ phân loại hành động (stgcnpp, lstm, gru hoặc tcn)."""
from patientmon import cli
from patientmon.actions.models import MODEL_NAMES
from patientmon.actions.train import train_action


def _args(parser):
    parser.add_argument("--model", choices=MODEL_NAMES, help="ghi đè train_action.model trong cấu hình")


def main():
    cfg, args = cli.init(__doc__, _args)
    best = train_action(cfg, model_name=args.model)
    print(f"Checkpoint tốt nhất: {best}")


if __name__ == "__main__":
    main()
