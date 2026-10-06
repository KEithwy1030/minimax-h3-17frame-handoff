"""Write the decode_temporal slice comparison. The decode itself is run by run_gpu.py."""
from slice_plan import chunk_rows, comfy_chunks, official_chunks, temporal_constants


def _fmt_span(span):
    if not span:
        return ""
    return ", ".join(f"{a}-{b - 1}" for a, b in span)


def _rows_md(rows):
    lines = [
        "| 块 | latent t | 送入解码器的 token 数 | 解码帧数 | 丢掉的局部帧 | 写出的成片帧 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        dropped = ",".join(str(i) for i in row["dropped_local_frames"])
        lines.append(
            f"| {row['chunk']} | {row['latent_t'][0]}:{row['latent_t'][1]} | {row['tokens_into_decoder']} | {row['decoded_frames']} | {dropped} | {_fmt_span(row['output_frame_spans'])} |"
        )
    return lines


def write_decode_slicing(path, z_len, comfy_log, official_log, comfy_luma, official_luma, weight_note):
    constants = temporal_constants()
    c_pad, c_n = comfy_chunks(z_len, constants)
    o_pad, o_n = official_chunks(z_len, constants, isolated_token_num=0)
    c_rows, c_out = chunk_rows(z_len, c_pad, c_n, constants)
    o_rows, o_out = chunk_rows(z_len, o_pad, o_n, constants)
    comfy_tokens = [item["tokens"] for item in comfy_log]
    official_tokens = [item["tokens"] for item in official_log]
    plan_tokens = [row["tokens_into_decoder"] for row in c_rows]
    lines = [
        "# 两套 decode_temporal 的切片",
        "",
        weight_note,
        "",
        "常数来自 `clip_length=17`、`token_drop=3`、`vae_ratio_t=4`：",
        "",
        f"- tokens_chunk_size = {constants['tokens_chunk_size']}",
        f"- token_overlap = {constants['token_overlap']}",
        f"- frame_pre_padding = {constants['frame_pre_padding']}",
        f"- frame_overlap = {constants['frame_overlap']}",
        f"- 每个 token 解码 {constants['vae_ratio_t']} 帧，split_count = {constants['split_count']}",
        "",
        f"输入 latent 时间长度 {z_len}。",
        f"ComfyUI `_decode_temporal_chunks`：pad_tokens={c_pad}，num_chunks={c_n}，公式给出的成片帧数（不含最后裁掉的 pad 帧，本次 pad 为 {c_pad}）写出终点 {c_out}。",
        f"官方 `decode_temporal` 在 `isolated_first_frame=False`、`isolated_last_frame=False` 时：pad_tokens={o_pad}，num_chunks={o_n}，写出终点 {o_out}。",
        "",
        "丢掉的局部帧是每个解码片段内部、被 `frame_pre_padding` 切掉的下标。`j=1` 留下的帧先当作下一块的 overlap，只有最后一块才直接写到成片末尾。",
        "",
        "## ComfyUI 公式",
        "",
        *_rows_md(c_rows),
        "",
        "## 官方公式（无 isolated frame）",
        "",
        *_rows_md(o_rows),
        "",
        "## 实际送进 `_adaptive_decode` 的 token 数",
        "",
        f"ComfyUI 调用顺序：{comfy_tokens or '没有调用'}",
        f"官方调用顺序：{official_tokens or '没有调用'}",
        f"公式中的 token 数：{plan_tokens}",
        "",
    ]
    if comfy_tokens == plan_tokens:
        lines.append("ComfyUI 每次实际送入的 token 数和公式一致。")
    else:
        lines.append("ComfyUI 每次实际送入的 token 数和公式不一致。")
    if official_tokens == plan_tokens:
        lines.append("官方每次实际送入的 token 数和公式一致。")
    elif official_tokens:
        lines.append("官方每次实际送入的 token 数和公式不一致。")
    else:
        lines.append("官方解码没有跑完，没有实际 token 数。")
    lines.append("")
    if comfy_log:
        lines.append("ComfyUI 每次解码输出帧数：" + ", ".join(str(item["out_frames"]) for item in comfy_log))
    if official_log:
        lines.append("官方每次解码输出帧数：" + ", ".join(str(item["out_frames"]) for item in official_log))
    lines.extend(["", "## 逐帧亮度", ""])
    if comfy_luma is None or official_luma is None:
        lines.append("有一侧没有亮度，不能做差。")
    else:
        a = [float(v) for v in comfy_luma]
        b = [float(v) for v in official_luma]
        n = min(len(a), len(b))
        lines.append(f"ComfyUI 成片 {len(a)} 帧，官方成片 {len(b)} 帧。下表比较前 {n} 帧。Y 是全帧 BT.601 平均值，0–255。")
        lines.append("")
        lines.append("| frame | comfy | official | official-comfy |")
        lines.append("| --- | --- | --- | --- |")
        for i in range(n):
            lines.append(f"| {i} | {a[i]:.4f} | {b[i]:.4f} | {b[i] - a[i]:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
