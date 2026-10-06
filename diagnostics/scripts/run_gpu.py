"""Run experiments 2–5 on one GPU load. Does not edit ComfyUI or pixels."""
import gc
import json
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
COMFY = Path(r"D:\MinmaxH3\ComfyUI")
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(COMFY))
os.chdir(COMFY)

import torch

import comfy.sd
from comfy.ldm.minimax.vae import IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD

from decode_slicing import write_decode_slicing
from decode_with_context import decode_with_context
from measure_io import (
    FP16_VAE,
    LATENT_PATH,
    OFFICIAL_VAE,
    OFFICIAL_VAE_BYTES,
    RESULTS,
    SEAM_FRAMES,
    dark_indices,
    deltas,
    frame_luma,
    seam_lines,
    write_luma_csv,
)
from roundtrip_static import run_roundtrip

OFFICIAL_SRC = Path(r"D:\MinmaxH3\runs\official_vae")
SOURCE_CFG = OFFICIAL_SRC / "source" / "config.json"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_video_latent():
    obj = torch.load(LATENT_PATH, map_location="cpu", weights_only=False)
    samples = obj["samples"]
    z = samples.tensors[0] if hasattr(samples, "tensors") else samples[0]
    if tuple(z.shape) != (1, 24, 37, 38, 66):
        raise RuntimeError(f"unexpected latent shape {tuple(z.shape)}")
    return z


def load_half_state(path):
    from safetensors import safe_open

    state = {}
    skipped = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key in ("latents_mean", "latents_std"):
                skipped.append(key)
                continue
            tensor = handle.get_tensor(key)
            state[key] = tensor if tensor.dtype == torch.float16 else tensor.half()
    return state, skipped


def wait_for_weight():
    deadline = time.time() + 20 * 60
    last = -1
    still = 0
    while time.time() < deadline:
        size = OFFICIAL_VAE.stat().st_size if OFFICIAL_VAE.exists() else 0
        log(f"official weight bytes {size} / {OFFICIAL_VAE_BYTES}")
        if size == OFFICIAL_VAE_BYTES:
            return OFFICIAL_VAE, (
                f"两套解码用的是同一份权重：`{OFFICIAL_VAE}`（官方 "
                "Ref2VA/video_vae/source/model.safetensors，10415548320 字节）。"
                "读入时逐张量从 fp32 转成 fp16，因为 RTX 3060 12GB 放不下 fp32 解码。"
            )
        if size == last:
            still += 1
            if still >= 4 and size != OFFICIAL_VAE_BYTES:
                break
        else:
            still = 0
            last = size
        time.sleep(15)
    return FP16_VAE, (
        f"官方 `model.safetensors` 没有下完整（停在 {last} 字节，目标 {OFFICIAL_VAE_BYTES}）。"
        f"实验 2 和实验 3 改用本地 `{FP16_VAE}`。"
    )


def hook_decode(model, bucket):
    real = model._adaptive_decode

    def wrapped(z):
        tokens = int(z.shape[2])
        out = real(z)
        bucket.append({"tokens": tokens, "out_frames": int(out.shape[2])})
        log(f"adaptive_decode #{len(bucket)} tokens={tokens} out_frames={out.shape[2]}")
        return out

    model._adaptive_decode = wrapped
    return real


def key_gap(module, state):
    module_keys = set(module.state_dict().keys())
    file_keys = set(state.keys())
    lack = sorted(k for k in module_keys - file_keys if k not in ("latents_mean", "latents_std", "pixel_mean", "pixel_std"))
    extra = sorted(file_keys - module_keys)
    return lack, extra


def load_comfy(state):
    cloned = {k: v.clone() for k, v in state.items()}
    vae = comfy.sd.VAE(sd=cloned, dtype=torch.float16)
    model = vae.first_stage_model
    model.cuda().eval()
    return vae, model


def free_module(*objs):
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def official_pixels(raw):
    mean = torch.tensor(IMAGENET_MEAN, device=raw.device, dtype=torch.float32).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=raw.device, dtype=torch.float32).view(1, 3, 1, 1, 1)
    return (raw.float() * std + mean).clamp_(0.0, 1.0)


def load_official(state):
    sys.path.insert(0, str(OFFICIAL_SRC.parent))
    from official_vae.klvae import AutoencoderKLLegacy

    model, _unused = AutoencoderKLLegacy.from_config(
        AutoencoderKLLegacy.load_config(str(SOURCE_CFG)),
        return_unused_kwargs=True,
        clip_length=17,
        token_drop=3,
        encoder_tiling=1,
        decoder_tiling=1,
        parallel_tiling=0,
        tile_size=256,
        tile_overlap_min=64,
        encoder_parallel=0,
        decoder_parallel=0,
        chunk_dim=-1,
    )
    model = model.half().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.cuda()
    return model, [str(k) for k in missing], [str(k) for k in unexpected]


def denorm_with_lists(z):
    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float16, device=z.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=torch.float16, device=z.device).view(1, -1, 1, 1, 1)
    return z.half() * std + mean


def write_context_report(stock, context, plan):
    lines = [
        "# 带上下文的分块解码",
        "",
        "没有改 `D:\\MinmaxH3\\ComfyUI\\comfy\\ldm\\minimax\\vae.py`。实验代码在 `diagnostics/scripts/decode_with_context.py`。",
        "每个块额外带上前一块最后 2 个 latent token，解码后丢掉这 2 个 token 对应的帧，当前块仍按原版的 frame_pre_padding 和 overlap 写出。第 0 块前面没有 token，不带上下文。",
        "权重、latent、空间范围和 ComfyUI 原版 `decode` 相同。",
        "",
        "## 原版，指定帧相对前一帧",
        "",
        *seam_lines(stock),
        "",
        "## 带 2 token 上下文，指定帧相对前一帧",
        "",
        *seam_lines(context),
        "",
        f"原版暗帧（相对前一帧 ≤ -15%）：{dark_indices(stock) or '无'}。",
        f"带上下文暗帧（相对前一帧 ≤ -15%）：{dark_indices(context) or '无'}。",
        "",
        "上下文计划（latent 下标是补 pad 之后的下标）：",
        "",
    ]
    for row in plan:
        lines.append(
            f"- 块 {row['chunk']}: 上下文 t {row['context_latent_t']}，当前 t {row['current_latent_t']}，"
            f"送入 token {row['tokens_into_decoder']}，丢掉预热帧 {row['warmup_frames_discarded']}，"
            f"写出 {row['output_frame_spans']}"
        )
    lines.extend(["", "逐帧表：`decode_with_context_luma.csv`。", ""])
    (RESULTS / "decode_with_context.md").write_text("\n".join(lines), encoding="utf-8")
    a = [float(v) for v in stock]
    b = [float(v) for v in context]
    n = min(len(a), len(b))
    da = deltas(a)
    db = deltas(b)
    import csv

    with (RESULTS / "decode_with_context_luma.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "luma_stock", "luma_context", "delta_stock", "delta_context"])
        for i in range(n):
            writer.writerow(
                [
                    i,
                    f"{a[i]:.6f}",
                    f"{b[i]:.6f}",
                    "" if da[i] is None else f"{da[i]:.8f}",
                    "" if db[i] is None else f"{db[i]:.8f}",
                ]
            )


def seam_snapshot(luma):
    if luma is None:
        return None
    values = [float(v) for v in luma]
    change = deltas(values)
    out = []
    for i in SEAM_FRAMES:
        if i >= len(values):
            continue
        out.append(
            {
                "frame": i,
                "luma": values[i],
                "delta_vs_prev": None if i == 0 else change[i],
            }
        )
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    free_b, total_b = torch.cuda.mem_get_info()
    log(f"free vram {free_b / 1024**2:.0f} MiB / {total_b / 1024**2:.0f}")
    if free_b < 8 * 1024**3:
        raise SystemExit("GPU 空闲显存不到 8GB，先停掉占用显存的 ComfyUI 再跑。")

    weight_path, weight_note = wait_for_weight()
    log(weight_note)
    state, skipped = load_half_state(weight_path)
    kept = {k: v.clone() for k, v in state.items()}
    del state
    gc.collect()

    z = load_video_latent()
    facts = {
        "weight_path": str(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "weight_note": weight_note,
        "skipped_keys": skipped,
        "latent_shape": list(z.shape),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": "float16",
    }

    log("load comfy vae")
    vae, model = load_comfy(kept)
    lack, extra = key_gap(model, kept)
    facts["comfy_missing_keys"] = lack[:30]
    facts["comfy_missing_count"] = len(lack)
    facts["comfy_extra_count"] = len(extra)
    facts["comfy_param_dtype"] = str(next(model.parameters()).dtype)
    if len(lack) > 20:
        raise SystemExit(f"comfy load missing {len(lack)} keys, first {lack[:8]}")

    z_gpu = z.half().cuda()
    comfy_log = []
    real = hook_decode(model, comfy_log)
    log("comfy decode_temporal full spatial")
    with torch.inference_mode():
        pix = model.decode(z_gpu)
    model._adaptive_decode = real
    comfy_luma = frame_luma(pix).cpu()
    del pix
    torch.cuda.empty_cache()
    log(f"comfy frames {comfy_luma.shape[0]}")

    log("context decode")
    with torch.inference_mode():
        ctx_pix, ctx_plan = decode_with_context(model, denorm_with_lists(z_gpu), context_tokens=2)
    ctx_luma = frame_luma(ctx_pix).cpu()
    del ctx_pix
    torch.cuda.empty_cache()
    write_context_report(comfy_luma, ctx_luma, ctx_plan)
    facts["context_seams"] = seam_snapshot(ctx_luma)
    facts["comfy_seams"] = seam_snapshot(comfy_luma)

    free_module(model, vae)
    log("freed comfy vae")

    official_luma = None
    official_log = []
    official_error = None
    try:
        log("load official vae")
        official, missing, unexpected = load_official({k: v.clone() for k, v in kept.items()})
        facts["official_missing_count"] = len(missing)
        facts["official_unexpected_count"] = len(unexpected)
        facts["official_missing_head"] = missing[:20]
        facts["official_clip_length"] = int(official.clip_length)
        facts["official_token_drop"] = int(official.token_drop)
        facts["official_decoder_tiling"] = bool(official.decoder_tiling)
        if len(missing) > 20:
            raise RuntimeError(f"official missing {len(missing)} keys")
        real = hook_decode(official, official_log)
        log("official decode_temporal full spatial")
        with torch.inference_mode():
            raw = official.decode_temporal(denorm_with_lists(z_gpu))
        official._adaptive_decode = real
        official_luma = frame_luma(official_pixels(raw)).cpu()
        del raw, official
        torch.cuda.empty_cache()
        log(f"official frames {official_luma.shape[0]}")
    except Exception as exc:
        official_error = f"{type(exc).__name__}: {exc}"
        log("official failed " + official_error)
        facts["official_error"] = official_error

    facts["official_seams"] = seam_snapshot(official_luma)
    facts["comfy_tokens"] = comfy_log
    facts["official_tokens"] = official_log

    write_decode_slicing(
        RESULTS / "decode_slicing.md",
        z_len=37,
        comfy_log=comfy_log,
        official_log=official_log,
        comfy_luma=comfy_luma,
        official_luma=official_luma,
        weight_note=weight_note,
    )
    if official_luma is not None:
        write_luma_csv(RESULTS / "frame_luma_full_decode.csv", official_luma)
        facts["full_decode_source"] = "official AutoencoderKLLegacy.decode_temporal"
    else:
        write_luma_csv(RESULTS / "frame_luma_full_decode.csv", comfy_luma)
        facts["full_decode_source"] = "comfy MiniMaxH3VideoVAE.decode because official decode failed: " + str(official_error)

    (RESULTS / "run_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    log("wrote decode tables")

    log("reload comfy vae for roundtrip")
    vae, model = load_comfy(kept)
    del kept
    gc.collect()
    try:
        with torch.inference_mode():
            facts["roundtrip"] = run_roundtrip(model, RESULTS)
    except Exception as exc:
        facts["roundtrip"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        log("roundtrip failed " + facts["roundtrip"]["error"])
        torch.cuda.empty_cache()
    free_module(model, vae)
    (RESULTS / "run_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    log("gpu experiments done")


if __name__ == "__main__":
    main()
