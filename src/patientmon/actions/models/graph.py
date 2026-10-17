"""Đồ thị khớp COCO-17 và phép chia vùng không gian của ST-GCN (chỉ dùng numpy)."""
from __future__ import annotations

import numpy as np

NUM_JOINTS = 17
CENTER = 0  # mũi
# Cây xương COCO-17 theo PYSKL: vai nối với mũi, hông nối với vai (không có cạnh vai-vai hay hông-hông)
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
# cặp khớp trái/phải, dùng khi lật ngang
LEFT_RIGHT = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _adjacency(num: int, edges) -> np.ndarray:
    adj = np.zeros((num, num))
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    return adj


def hop_distance(num: int, edges) -> np.ndarray:
    adj = _adjacency(num, edges)
    hop = np.full((num, num), np.inf)
    np.fill_diagonal(hop, 0)
    step, reach = adj + np.eye(num), np.eye(num)
    for d in range(1, num):
        reach = (reach @ step) > 0
        hop[reach & np.isinf(hop)] = d
    return hop


def bone_parents(num: int = NUM_JOINTS, edges=COCO_EDGES, root: int = CENTER) -> list:
    """Cha của mỗi khớp trong cây bao trùm tính từ khớp gốc (gốc tự trỏ vào mình)."""
    parent, seen, queue = [root] * num, {root}, [root]
    adj = _adjacency(num, edges)
    while queue:
        u = queue.pop(0)
        for v in np.nonzero(adj[u])[0]:
            if int(v) not in seen:
                seen.add(int(v))
                parent[int(v)] = u
                queue.append(int(v))
    return parent


def spatial_adjacency(num: int = NUM_JOINTS, edges=COCO_EDGES, center: int = CENTER) -> np.ndarray:
    """Ba tập con (K=3) theo ST-GCN: (0) chính nó, (1) láng giềng ngang hoặc xa tâm hơn, (2) láng giềng gần tâm hơn;
    chuẩn hóa theo bậc. Hình dạng (3, V, V)."""
    hop = hop_distance(num, edges)
    adj = _adjacency(num, edges) + np.eye(num)
    norm = adj / np.maximum(adj.sum(axis=0, keepdims=True), 1e-9)
    parts = np.zeros((3, num, num))
    for i in range(num):
        for j in range(num):
            if hop[j, i] == 0:
                parts[0, j, i] = norm[j, i]
            elif hop[j, i] == 1:
                parts[1 if hop[j, center] >= hop[i, center] else 2, j, i] = norm[j, i]
    return parts
