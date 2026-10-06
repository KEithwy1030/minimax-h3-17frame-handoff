"""Shared numeric writers. No pixel modification."""
import csv
from pathlib import Path

import torch

RESULTS = Path(r"D:\MinmaxH3\h3-17frame-handoff\diagnostics\results")
FP16_VAE = Path(r"D:\MinmaxH3\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors")
OFFICIAL_VAE = Path(r"D:\MinmaxH3\runs\official_fp32_vae\model.safetensors")
OFFICIAL_VAE_BYTES = 10415548320
LATENT_PATH = Path(r"D:\MinmaxH3\ComfyUI\output\minimax_seg_cache\12\seg_0000.av.pt")
STILL_PATH = Path(r"D:\MinmaxH3\ComfyUI\output\h3_watch_stock_00001_.png")
SEAM_FRAMES = (0, 17, 34, 51, 68, 85, 102, 119)


def frame_luma(video):
    """video [1,3,T,H,W] in 0..1. Returns a CPU float tensor of length T, Y in 0..255."""
    v = video.detach()
    if v.device.type != "cpu":
        v = v.float().cpu()
    else:
        v = v.float()
    y = 0.299 * v[0, 0] + 0.587 * v[0, 1] + 0.114 * v[0, 2]
    return y.mean(dim=(-2, -1)).mul(255.0)


def deltas(luma):
    out = [None]
    for i in range(1, len(luma)):
        prev = float(luma[i - 1])
        out.append((float(luma[i]) - prev) / prev if prev != 0 else None)
    return out


def write_luma_csv(path, luma, extra_columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(x) for x in luma]
    dlt = deltas(values)
    fieldnames = ["frame", "luma", "delta_vs_prev"]
    if extra_columns:
        fieldnames.extend(extra_columns.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, y in enumerate(values):
            row = {
                "frame": i,
                "luma": f"{y:.6f}",
                "delta_vs_prev": "" if dlt[i] is None else f"{dlt[i]:.8f}",
            }
            if extra_columns:
                for key, seq in extra_columns.items():
                    item = seq[i]
                    row[key] = "" if item is None else item
            w.writerow(row)
    return values, dlt


def seam_lines(luma):
    values = [float(x) for x in luma]
    dlt = deltas(values)
    lines = []
    for i in SEAM_FRAMES:
        if i >= len(values):
            lines.append(f"- 帧 {i}：这次输出只有 {len(values)} 帧，没有这一帧。")
            continue
        if i == 0 or dlt[i] is None:
            lines.append(f"- 帧 {i}：亮度 {values[i]:.2f}。没有前一帧，不算相对变化。")
        else:
            lines.append(
                f"- 帧 {i} 相对帧 {i - 1}：{values[i - 1]:.2f} → {values[i]:.2f}，变化 {dlt[i] * 100:.2f}%。"
            )
    return lines


def dark_indices(luma, threshold=-0.15):
    values = [float(x) for x in luma]
    dlt = deltas(values)
    found = []
    for i in range(1, len(values)):
        if dlt[i] is not None and dlt[i] <= threshold:
            found.append(i)
    return found


def denorm_latent(z, mean, std):
    mean_t = torch.tensor(mean, dtype=torch.float32, device=z.device).view(1, -1, 1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=z.device).view(1, -1, 1, 1, 1)
    return z.float() * std_t + mean_t
