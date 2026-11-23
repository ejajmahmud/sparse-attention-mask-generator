"""CLI for Sparse Attention Mask Generator."""
import argparse
from .core import SparseMaskGenerator

def main():
    parser = argparse.ArgumentParser(description="Sparse Attention Mask Generator CLI")
    parser.add_argument("--demo", action="store_true", help="Run sparse mask generation demo")
    args = parser.parse_args()

    layout = SparseMaskGenerator.generate_block_sparse_mask(seq_len=8, local_window_size=2, global_anchor_indices=[0])
    print(f"Generated 8x8 Mask with Sparsity: {layout.sparsity_ratio * 100:.1f}%")
    for row in layout.mask_grid:
        print("  " + " ".join(str(x) for x in row))

if __name__ == "__main__":
    main()
