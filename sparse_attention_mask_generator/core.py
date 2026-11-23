"""Core Sparse Attention Mask Generator implementation."""
from typing import List, Tuple

class MaskLayout:
    def __init__(self, size: int, sparsity_ratio: float, mask_grid: List[List[int]]):
        self.size = size
        self.sparsity_ratio = sparsity_ratio
        self.mask_grid = mask_grid

class SparseMaskGenerator:
    @staticmethod
    def generate_block_sparse_mask(
        seq_len: int,
        local_window_size: int = 4,
        global_anchor_indices: Optional[List[int]] = None
    ) -> MaskLayout:
        anchors = set(global_anchor_indices or [0])
        grid = [[0 for _ in range(seq_len)] for _ in range(seq_len)]
        ones_count = 0

        for i in range(seq_len):
            for j in range(i + 1):  # Causal mask (j <= i)
                # 1. Global anchors: always attend to anchors or if i is anchor
                if j in anchors or i in anchors:
                    grid[i][j] = 1
                    ones_count += 1
                # 2. Local sliding window
                elif (i - j) <= local_window_size:
                    grid[i][j] = 1
                    ones_count += 1

        total_causal = (seq_len * (seq_len + 1)) // 2
        sparsity = 1.0 - (ones_count / total_causal) if total_causal > 0 else 0.0

        return MaskLayout(seq_len, round(sparsity, 4), grid)
