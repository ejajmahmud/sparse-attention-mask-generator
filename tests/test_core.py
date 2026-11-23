"""Tests for SparseMaskGenerator."""
import unittest
from sparse_attention_mask_generator.core import SparseMaskGenerator

class TestSparseMaskGenerator(unittest.TestCase):
    def test_mask_generation(self):
        m = SparseMaskGenerator.generate_block_sparse_mask(seq_len=6, local_window_size=2, global_anchor_indices=[0])
        self.assertEqual(m.size, 6)
        self.assertEqual(len(m.mask_grid), 6)
        # Anchor at index 0 must be 1 for all rows
        for i in range(6):
            self.assertEqual(m.mask_grid[i][0], 1)
        self.assertGreater(m.sparsity_ratio, 0.0)
