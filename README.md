# MiniMax H3：每 17 帧暗一帧

给外部 agent 的交接。只保留已测量的结论和复现所需路径。不要再做整帧补亮度或淡入淡出。

## 结论

暗帧在 **ComfyUI 生成出来的 latent** 里。用 MiniMax 官方 VAE 源码、加载本地同一套 VAE 权重，解码这份 latent，第 17 帧仍然暗约 36%。

因此这不是「ComfyUI 独有的拼帧算法」单独能解释的。官方解码函数解同一份 latent，结果同样暗。

MiniMax 自己发布的样片 `t2va_2k.mp4` 没有这个周期（见 ComfyUI issue #15426）。那是他们整套程序生成的片子，不是拿下面这份 latent 去解。

## 现象

- 24 fps，大约每 0.7 秒暗一帧。
- 暗帧下标（从 0 计）：0、17、34、51、68、85、102、119。
- 第 0 帧也偏暗。周期从 0 开始，不是从 17 才开始。
- 暗的是单独峰值帧，下一帧会回一截，但第 17 帧相对第 16 帧大约暗 30%–36%。

## 已测量

同一 latent：`D:\MinmaxH3\ComfyUI\output\minimax_seg_cache\12\seg_0000.av.pt`  
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
本地副本：`D:\MinmaxH3\runs\official_vae\`（只有源码，没有 10GB 的 `source/model.safetensors`）。

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

## 还没分开的一点

静止图如果 **整段一次编码**（不按 17 帧切开）再解码，只有第 0 帧偏暗，第 17 帧不再暗。按 17 帧分段编码再解码，第 17 帧会暗。

生成出来的 latent 用官方 `decode_temporal` 解，表现和「分段编码」一样。还没有用 MiniMax 官方 10GB `Ref2VA/video_vae/source/model.safetensors` 重解；这次用的是 Comfy 版 fp16 权重，键和官方类一致。

## 代码位置

ComfyUI：`D:\MinmaxH3\ComfyUI`，commit `8a33128`（`Improve some warning messages. (#15977)`）。  
`comfy/ldm/minimax/vae.py` 的 `decode_temporal` 当前与该 commit 一致，没有本地补丁。

模型里 17 帧一组的定义：`comfy/ldm/minimax/model.py` 的 `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`。  
完整版权重才走 `t_emb = self.time_embedder(t_vals).to(dtype)`（约第 717 行）。pruned 走 `use_adaln_curves` 查表，不走这一行。两边生成结果仍然同样暗。

## GitHub

- 同一现象，仍开放，无已验证修复：<https://github.com/Comfy-Org/ComfyUI/issues/15426>
- 完整版 fl2va 权重的另一报告（未覆盖本次 ref2va 对照）：<https://github.com/MiniMax-AI/MiniMax-H3/issues/85>
- 遮罩格子，不是这次的问题：<https://github.com/Comfy-Org/ComfyUI/issues/15978>

## 建议的下一步

不要再改像素。拿官方 `decode_temporal` 和 ComfyUI `decode_temporal` 对 **同一 latent、同一权重** 逐项对比切帧下标。官方源码已在 `D:\MinmaxH3\runs\official_vae\klvae.py` 的 `decode_temporal`。若两套切帧一致且都暗，就去查采样产出的 latent，而不是解码拼帧。
