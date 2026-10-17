"""Nhãn một-cho-cả-đoạn lấy từ ID của nguồn (đường dẫn viết không dấu, nối bằng "__")."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional


class Labeler:
    def __init__(self, classes: List[str], rules: List[Dict[str, str]]):
        if not rules:
            raise ValueError(
                "Chưa có action.label_rules trong file cấu hình. Hãy khai báo cách suy ra nhãn từ tên thư mục/file, "
                'ví dụ: [{pattern: "walk", label: walk}].'
            )
        self.classes = list(classes)
        self.rules = []
        for rule in rules:
            if rule["label"] not in self.classes:
                raise ValueError(f"Nhãn '{rule['label']}' không nằm trong action.classes {self.classes}")
            self.rules.append((re.compile(rule["pattern"], re.IGNORECASE), self.classes.index(rule["label"])))

    def label_for(self, source_id: str) -> Optional[int]:
        """Chỉ số lớp của luật đầu tiên khớp; None nếu không luật nào khớp."""
        for pattern, label in self.rules:
            if pattern.search(source_id):
                return label
        return None


def make_group_fn(pattern: Optional[str]) -> Callable[[str], str]:
    """Hàm lấy mã nhóm (người) từ ID nguồn; không khớp hoặc không có pattern thì mỗi nguồn là một nhóm."""
    compiled = re.compile(pattern, re.IGNORECASE) if pattern else None

    def group_of(source_id: str) -> str:
        if compiled:
            m = compiled.search(source_id)
            if m:
                return m.group(1)
        return source_id

    return group_of
