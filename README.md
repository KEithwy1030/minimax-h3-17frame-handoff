# MiniMax H3：每 17 帧暗一帧

给外部 agent 的交接。只保留已测量的结论和复现所需路径。不要再做整帧补亮度或淡入淡出。

## 结论

暗帧来自本地 fp16 VAE 权重文件。官方权重解同一 latent 不暗。

- A：`D:\MinmaxH3\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors`，5207808496 字节。这是原来会闪的 ComfyUI 视频 VAE。
- B：`D:\MinmaxH3\runs\official_fp32_vae\model.safetensors`，10415548320 字节。来自 Hugging Face `MiniMaxAI/MiniMax-H3` 的 `Ref2VA/video_vae/source/model.safetensors`。

B 先转成 fp16 再和 A 比。560 个共有键里 557 个完全相同。有差异的只有 3 个：`decoder.transformer_blocks.0.attn.to_out.bias`、`decoder.transformer_blocks.0.attn.to_out.weight`、`decoder.register_tokens`。A 另外多了 `latents_mean` / `latents_std`，它们和 `vae.py` 里写死的数只差 fp16 舍入。明细在 `diagnostics/results/vae_weight_diff.md`。

用 B 转成的 fp16，加上 `vae.py` 里的 `latents_mean` / `latents_std`，保存为 `D:\MinmaxH3\ComfyUI\models\vae\minimax_h3_video_vae_official_fp16.safetensors`（5207808592 字节，没有放进仓库）。工作流只换这个 VAE：

- 22 帧、640×384、pruned ref2va、种子 1：第 15–19 帧亮度 57.70、57.88、57.50、57.75、57.83。第 16→17 帧下降 0.66%。
- 124 帧、864×480、同一套采样设置：第 17/34/51/68/85/102/119 帧相对前一帧的变化是 -0.63%、-0.59%、-0.70%、-0.80%、-0.61%、-0.59%、-0.50%。

会闪运行自己的 latent 用本地文件解码时，第 16→17 帧是 61.01 → 42.44，下降 30.4%。`seg_0000.av.pt` 不是那份 latent。MiniMax 样片 `t2va_2k.mp4` 没有这个周期（见 ComfyUI issue #15426）。

## 现象

- 24 fps，大约每 0.7 秒暗一帧。
- 暗帧下标（从 0 计）：0、17、34、51、68、85、102、119。
- 第 0 帧也偏暗。周期从 0 开始，不是从 17 才开始。
- 暗的是单独峰值帧，下一帧会回一截，但第 17 帧相对第 16 帧大约暗 30%–36%。

## 已测量

下面这张表不是会闪运行的 latent。文件是 `D:\MinmaxH3\ComfyUI\output\minimax_seg_cache\12\seg_0000.av.pt`  
形状：`samples` 视频支路 `[1, 24, 37, 38, 66]` float32。对应成片约 1056×608、124 帧。

用官方 `AutoencoderKLLegacy.decode_temporal`（`clip_length=17`，`token_drop=3`），空间只取中心 8×8 latent（128×128 像素），权重用本地 `minimax_h3_video_vae_fp16.safetensors`（去掉 `latents_mean` / `latents_std` 后 `load_state_dict` 缺失 0、多余 0）：

| 帧 | 亮度 0–255 | 相对前一帧 |
| --- | --- | --- |
| 16 → 17 | 108 → 69 | −36.0% |
| 33 → 34 | 107 → 68 | −36.7% |
| 50 → 51 | 109 → 69 | −36.6% |
| 67 → 68 | 124 → 80 | −35.7% |
| 84 → 85 | 114 → 74 | −35.4% |
| 101 → 102 | 113 → 71 | −36.7% |
| 118 → 119 | 111 → 72 | −35.3% |

官方源码：`https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/Ref2VA/video_vae`  
本地源码副本：`D:\MinmaxH3\runs\official_vae\`。官方权重在 `D:\MinmaxH3\runs\official_fp32_vae\model.safetensors`（10415548320 字节），没有放进仓库。

## 对照：瘦身版和完整版都会闪

ComfyUI 原始解码（无补亮度、无淡入淡出）。同一张参考图 `ComfyUI_00005_.png`，同一提示、种子 1、8 步、`res_multistep` + `simple`、shift 12/3、640×384、22 帧。

| 权重 | 第 16→17 帧亮度跌幅 | 进入第 17 帧的像素差 / 正常换帧中位数 |
| --- | --- | --- |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 30.7% | 8.65× |
| `minimax_h3_ref2va_int8_convrot.safetensors` | 30.8% | 8.65× |

所以「换成 pruned 就不闪」对这次的 ref2va 不成立。`fl2va` 是首尾帧模型，不是这条参考图工作流。

可看的未修补成片（原始解码）：`D:\MinmaxH3\ComfyUI\output\video\h3_watch_stock.mp4`  
864×480，124 帧，12 步，pruned ref2va。第 17 帧及之后每个 17 帧边界约暗 34%。

## 已排除

- 帧数不合法。124 和 22 都是 `17k+5`（除以 17 余 5），仍然闪。
- 步数。ComfyUI [#15426](https://github.com/Comfy-Org/ComfyUI/issues/15426) 在官方模板上试过 20/30/40 步，仍然闪。这次 8 步和 12 步也闪。
- 瘦身版 vs 完整版 ref2va。见上表，跌幅相同。
- 去噪遮罩 / PR #15988。这次生成没有使用 denoise mask。那次修复解决的是遮罩重绘出格子，不是无遮罩时每 17 帧变暗。
- 空间 tiled decode 节点。测试用的是普通 `VAEDecode`。H3 VAE 内部仍会做时间分块。
- 整帧乘一个亮度、或把接缝 4 帧做淡入淡出。前者会让接缝局部过亮，后者肉眼是渐变。都不要再做。

## 已经分开的一点

会闪运行自己的 latent，用官方 10GB 权重整段解码，第 17 帧没有大约 30% 的下跌。用本地 fp16 权重解同一份 latent，下跌还在。`seg_0000` 和这次 latent 不是同一份。

## 代码位置

ComfyUI：`D:\MinmaxH3\ComfyUI`，commit `8a33128`（`Improve some warning messages. (#15977)`）。  
`comfy/ldm/minimax/vae.py` 的 `decode_temporal` 当前与该 commit 一致，没有本地补丁。

模型里 17 帧一组的定义：`comfy/ldm/minimax/model.py` 的 `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`。  
完整版权重才走 `t_emb = self.time_embedder(t_vals).to(dtype)`（约第 717 行）。pruned 走 `use_adaln_curves` 查表，不走这一行。当时两边都用本地 fp16 VAE，生成结果同样暗。

## GitHub

- 同一现象，仍开放，无已验证修复：<https://github.com/Comfy-Org/ComfyUI/issues/15426>
- 完整版 fl2va 权重的另一报告（未覆盖本次 ref2va 对照）：<https://github.com/MiniMax-AI/MiniMax-H3/issues/85>
- 遮罩格子，不是这次的问题：<https://github.com/Comfy-Org/ComfyUI/issues/15978>

## 建议的下一步

生成时用 `minimax_h3_video_vae_official_fp16.safetensors`。不要再改像素，也不要补亮度或做淡入淡出。
