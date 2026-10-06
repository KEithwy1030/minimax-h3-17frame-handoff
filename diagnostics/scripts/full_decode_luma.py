"""Experiment 3 writes results/frame_luma_full_decode.csv.

run_gpu.py loads the weights once and calls measure_io.write_luma_csv.
Columns: frame, luma, delta_vs_prev.
luma is the full-frame BT.601 Y mean on a 0–255 scale.
The spatial latent is the full 37x38x66 tensor, not a center crop.
"""
from measure_io import RESULTS, write_luma_csv


def write_full_decode(luma):
    return write_luma_csv(RESULTS / "frame_luma_full_decode.csv", luma)
