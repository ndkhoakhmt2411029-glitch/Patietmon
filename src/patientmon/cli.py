"""Phần khởi tạo chung cho các script dòng lệnh."""
from __future__ import annotations

import argparse
import logging

from .config import Config


def init(description, add_args=None):
    """Đọc tham số dòng lệnh, bật log, nạp cấu hình. Trả về (cfg, args)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/config.yaml")
    if add_args:
        add_args(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return Config.from_yaml(args.config), args
