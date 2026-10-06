"""Per-time statistics of the saved MiniMax H3 video latent.

Does not decode, does not change pixels, does not write the latent.
"""
import csv
import sys
from pathlib import Path

import torch

COMFY = Path(r"D:\MinmaxH3\ComfyUI")
sys.path.insert(0, str(COMFY))

from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD  # noqa: E402

LATENT_PATH = Path(r"D:\MinmaxH3\ComfyUI\output\minimax_seg_cache\12\seg_0000.av.pt")
OUT_DIR = Path(r"D:\MinmaxH3\h3-17frame-handoff\diagnostics\results")


def load_video_latent(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    samples = obj["samples"] if isinstance(obj, dict) else obj
    if hasattr(samples, "tensors"):
        z = samples.tensors[0]
    elif isinstance(samples, (list, tuple)):
        z = samples[0]
    else:
        z = samples
    if z.ndim == 4:
        z = z.unsqueeze(0)
    return z.float().contiguous()


def denormalize(z):
    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
    return z * std + mean


def stats_for_t(z):
    rows = []
    for t in range(z.shape[2]):
        x = z[0, :, t]
        rows.append(
            {
                "t": t,
                "t_mod_5": t % 5,
                "mean": float(x.mean()),
                "std": float(x.std(unbiased=False)),
                "abs_mean": float(x.abs().mean()),
                "l2": float(torch.linalg.vector_norm(x.reshape(-1), ord=2)),
            }
        )
    return rows


def group_rows(rows):
    groups = []
    for mod in range(5):
        part = [r for r in rows if r["t_mod_5"] == mod]
        def avg(key):
            return sum(r[key] for r in part) / len(part)
        groups.append(
            {
                "t_mod_5": mod,
                "count": len(part),
                "mean": avg("mean"),
                "std": avg("std"),
                "abs_mean": avg("abs_mean"),
                "l2": avg("l2"),
            }
        )
    return groups


def rel_gap(groups, key):
    base = groups[0][key]
    others = [g[key] for g in groups[1:]]
    other = sum(others) / len(others)
    if other == 0:
        return base, other, None
    return base, other, (base - other) / abs(other)


def write_csv(path, per_space):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["space", "t", "t_mod_5", "mean", "std", "abs_mean", "l2"]
        )
        w.writeheader()
        for space, rows in per_space:
            for row in rows:
                w.writerow({"space": space, **row})


def fmt(value):
    return f"{value:.8g}"


def write_md(path, shape, per_space, groups_by_space):
    lines = [
        "# latent 按时间位置的统计",
        "",
        f"文件：`{LATENT_PATH}`",
        f"视频支路形状：`{list(shape)}`，dtype 读入后按 float32 统计。",
        "标准差是总体标准差（`unbiased=False`）。L2 是该时间下标上 C/H/W 全部元素的欧氏范数。",
        "反归一化：`z * latents_std + latents_mean`，系数来自 `comfy/ldm/minimax/vae.py` 的 `LATENTS_MEAN` / `LATENTS_STD`。这是 `MiniMaxH3VideoVAE.decode` 送进解码器之前的值。",
        "",
    ]
    for space, groups in groups_by_space:
        lines.append(f"## {space}，按 t % 5 汇总（表内是该组各时间下标统计量的平均）")
        lines.append("")
        lines.append("| t%5 | 个数 | mean | std | abs_mean | L2 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for g in groups:
            lines.append(
                f"| {g['t_mod_5']} | {g['count']} | {fmt(g['mean'])} | {fmt(g['std'])} | {fmt(g['abs_mean'])} | {fmt(g['l2'])} |"
            )
        lines.append("")
        bits = []
        for key in ("mean", "std", "abs_mean", "l2"):
            base, other, gap = rel_gap(groups, key)
            if gap is None:
                bits.append(f"{key}: t%5==0 为 {fmt(base)}，其余组平均为 0")
            else:
                bits.append(
                    f"{key}: t%5==0 为 {fmt(base)}，其余四组平均为 {fmt(other)}，相对差 {gap * 100:.2f}%"
                )
        abs_base, abs_other, abs_gap = rel_gap(groups, "abs_mean")
        l2_base, l2_other, l2_gap = rel_gap(groups, "l2")
        obvious = abs(abs_gap or 0) > 0.05 or abs(l2_gap or 0) > 0.05
        if obvious:
            sentence = "t%5==0 这一组和其余组有差异（abs_mean 或 L2 的相对差超过 5%）。"
        else:
            sentence = "t%5==0 这一组和其余组没有明显差异（abs_mean 和 L2 的相对差都不超过 5%）。"
        lines.append(sentence + " " + "；".join(bits) + "。")
        lines.append("")
    lines.append("逐时间下标的原始数在 `latent_stats.csv`。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    z = load_video_latent(LATENT_PATH)
    if tuple(z.shape) != (1, 24, 37, 38, 66):
        raise SystemExit(f"unexpected shape {tuple(z.shape)}")
    raw = stats_for_t(z)
    den = stats_for_t(denormalize(z))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "latent_stats.csv", [("normalized", raw), ("denormalized", den)])
    write_md(
        OUT_DIR / "latent_stats.md",
        z.shape,
        [("normalized", raw), ("denormalized", den)],
        [("反归一化前", group_rows(raw)), ("反归一化后", group_rows(den))],
    )
    print("wrote", OUT_DIR / "latent_stats.csv", flush=True)


if __name__ == "__main__":
    main()
