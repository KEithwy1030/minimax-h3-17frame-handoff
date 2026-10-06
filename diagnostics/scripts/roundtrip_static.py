"""Encode and decode one repeated still. Does not modify ComfyUI's vae.py."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_with_context import decode_with_context
from measure_io import STILL_PATH, dark_indices, frame_luma, seam_lines, write_luma_csv


def repeated_still(side, frames=124):
    from PIL import Image
    import numpy as np

    image = Image.open(STILL_PATH).convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
    arr = torch.tensor(np.array(image, copy=True), dtype=torch.float32).div_(255.0)
    clip = arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, frames, 1, 1)
    return clip.mul(2).sub_(1).contiguous()


def _params(model):
    p = next(model.parameters())
    return p.device, p.dtype


def chunked_encode(model, x):
    device, dtype = _params(model)
    return model.encode(x.to(device=device, dtype=dtype), device=device)


def oneshot_roundtrip(model, x):
    device, dtype = _params(model)
    xin = x.to(device=device, dtype=dtype)
    moments = model._adaptive_encode(model._normalize_pixels(xin))
    mean = torch.chunk(moments.float(), 2, dim=1)[0]
    latents_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(mean)
    latents_std = model.latents_std.view(1, -1, 1, 1, 1).to(mean)
    z = (mean - latents_mean) / latents_std
    z_den = z * latents_std + latents_mean
    raw = model._adaptive_decode(z_den.to(dtype=dtype))
    return model._finalize_pixels(raw), int(z.shape[2])


def run_roundtrip(model, out_dir):
    out_dir = Path(out_dir)
    notes = [
        "256×256 的整段一次编码会占满 12GB 显存，跑了 20 分钟仍未返回。三种往返都改在 128×128 上做，帧数仍是 124。"
    ]
    chosen = None
    for side in (128,):
        torch.cuda.empty_cache()
        try:
            x = repeated_still(side)
            z = chunked_encode(model, x)
            z = z.to(dtype=next(model.parameters()).dtype)
            pix_chunk = model.decode(z)
            luma_chunk = frame_luma(pix_chunk)
            del pix_chunk
            mean = model.latents_mean.view(1, -1, 1, 1, 1).to(dtype=z.dtype, device=z.device)
            std = model.latents_std.view(1, -1, 1, 1, 1).to(dtype=z.dtype, device=z.device)
            pix_ctx, ctx_plan = decode_with_context(model, z * std + mean, context_tokens=2)
            luma_ctx = frame_luma(pix_ctx)
            del pix_ctx, z
            torch.cuda.empty_cache()
            pix_one, latent_t_one = oneshot_roundtrip(model, x)
            luma_one = frame_luma(pix_one)
            del pix_one, x
            torch.cuda.empty_cache()
            chosen = {
                "side": side,
                "luma_chunk": luma_chunk,
                "luma_ctx": luma_ctx,
                "luma_one": luma_one,
                "latent_t_one": latent_t_one,
                "ctx_plan": ctx_plan,
            }
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            notes.append(f"{side}x{side} 显存不足，三种往返都没有保留。")
            chosen = None
    if chosen is None:
        text = [
            "# 静止图往返",
            "",
            f"静止图：`{STILL_PATH}`，重复 124 帧。",
            "256 和 128 两种边长都在整段一次编码时显存不足，没有亮度表。",
            "",
        ]
        (out_dir / "roundtrip_static.md").write_text("\n".join(text), encoding="utf-8")
        return {"ok": False, "notes": notes}

    side = chosen["side"]
    write_luma_csv(out_dir / "roundtrip_oneshot.csv", chosen["luma_one"])
    write_luma_csv(out_dir / "roundtrip_chunked.csv", chosen["luma_chunk"])
    write_luma_csv(out_dir / "roundtrip_context.csv", chosen["luma_ctx"])
    lines = [
        "# 静止图往返",
        "",
        f"静止图：`{STILL_PATH}`，缩放到 {side}×{side} 后重复 124 帧。原图不是这个尺寸；256 放不下时才改用 128。",
        "像素范围按 VAE `encode` 的约定换成 [-1, 1]。没有改 ComfyUI 源文件。",
        "亮度是全帧 BT.601 Y 的平均值，0–255。",
        f"整段一次编码得到的 latent 时间长度：{chosen['latent_t_one']}。",
        "",
        "暗帧指相对前一帧亮度下降至少 15% 的帧。",
        "",
        "## 整段一次编码解码",
        "",
        f"暗帧下标：{dark_indices(chosen['luma_one']) or '无'}。",
        *seam_lines(chosen["luma_one"]),
        "",
        "## 按 17 帧分段编码，再按原版 decode_temporal 解码",
        "",
        f"暗帧下标：{dark_indices(chosen['luma_chunk']) or '无'}。",
        *seam_lines(chosen["luma_chunk"]),
        "",
        "## 按 17 帧分段编码，再用带 2 个前置 latent token 的解码",
        "",
        f"暗帧下标：{dark_indices(chosen['luma_ctx']) or '无'}。",
        *seam_lines(chosen["luma_ctx"]),
        "",
        "逐帧表：`roundtrip_oneshot.csv`、`roundtrip_chunked.csv`、`roundtrip_context.csv`。",
        "",
    ]
    if notes:
        lines.extend(["没有采用的尝试：", *[f"- {n}" for n in notes], ""])
    (out_dir / "roundtrip_static.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "ok": True,
        "side": side,
        "notes": notes,
        "latent_t_one": chosen["latent_t_one"],
        "dark_oneshot": dark_indices(chosen["luma_one"]),
        "dark_chunked": dark_indices(chosen["luma_chunk"]),
        "dark_context": dark_indices(chosen["luma_ctx"]),
        "n_oneshot": int(chosen["luma_one"].shape[0]),
        "n_chunked": int(chosen["luma_chunk"].shape[0]),
        "n_context": int(chosen["luma_ctx"].shape[0]),
    }
