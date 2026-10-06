# 这些源码分别来自哪里

路径按仓库里的原样记录。下面的文件都是完整原文，不是摘要。

## ComfyUI 本地代码

ComfyUI 仓库：`D:\MinmaxH3\ComfyUI`  
commit：`8a33128f2f8c5585c57486c07de481241e70a39c`

| 本诊断目录中的文件 | 本机原文件 |
| --- | --- |
| `ComfyUI/comfy/ldm/minimax/vae.py` | `D:\MinmaxH3\ComfyUI\comfy\ldm\minimax\vae.py` |
| `ComfyUI/comfy/ldm/minimax/model.py` | `D:\MinmaxH3\ComfyUI\comfy\ldm\minimax\model.py` |
| `ComfyUI/comfy/latent_formats.py` | `D:\MinmaxH3\ComfyUI\comfy\latent_formats.py` |

`latent_formats.py` 里的 `MiniMaxH3Video` 没有重写 `process_in` / `process_out`。基类 `LatentFormat.process_in` 只做 `latent * scale_factor`，而 `MiniMaxH3Video.scale_factor = 1.0`。这个类里也没有 `latents_mean` / `latents_std`。反归一化写在 `vae.py` 的 `MiniMaxH3VideoVAE.decode`：`z * latents_std + latents_mean`。

`sd_VAE_minimax_branches.py`、`model_base_minimax_latent.py`、`supported_models_MiniMaxH3.py` 是从对应文件按行号原样摘出的分支，行号写在每个片段前面。`comfy/sd.py` 的 `VAE` 类没有 `process_latent_in` / `process_latent_out`。这两个方法在 `comfy/model_base.py`。工作区里的 `comfy/sd.py` 比 commit `8a33128` 多 4 行：在 MiniMax H3 video VAE 分支里把 `disable_offload` 设成 `True`。摘录包含这 4 行。

## 官方 MiniMax-H3 仓库

仓库：https://github.com/MiniMax-AI/MiniMax-H3

官方仓库的 `model_index.json` 指向 diffusers 的 `MiniMaxH3ModularPipeline`，仓库里面没有单独的 Ref2VA / T2VA 推理 `.py` 去调用 `vae.decode`。

调用解码、并做 latent 反归一化的文件在 diffusers，不在 MiniMax-H3 仓库里：

- 上游路径：`huggingface/diffusers` `src/diffusers/modular_pipelines/minimax_h3/decoders.py`
- 类：`MiniMaxH3VideoDecodeStep`
- 取得方式：GitHub `main`，blob sha `bdc7adf7c5e729f5fb8b251ee00be45af9d16062`，10700 字节
- 本目录副本：`diffusers/modular_pipelines/minimax_h3/decoders.py`
- 反归一化在这个文件里：`latents * latents_std + latents_mean`，然后 `components.vae.decode(...)`
- 这次调用没有把 `clip_length` 或 `token_drop` 当参数传进去

`clip_length` / `token_drop` 在 VAE 配置里：

- 官方仓库 `Ref2VA/video_vae/config.json`：`vae_clip_length = 17`，`vae_token_drop = 3`。本地副本原来在 `D:\MinmaxH3\runs\official_vae\config.json`，这里是 `official/Ref2VA/video_vae/config.json`。
- 传参文件：官方仓库 `Ref2VA/video_vae/minimax_h3_video_vae.py` 的 `from_pretrained`，把上述两个值和 tile 设置传给 `AutoencoderKLLegacy.from_config`。这里是 `official/Ref2VA/video_vae/minimax_h3_video_vae.py`。
- 解码实现：官方仓库 `Ref2VA/video_vae/klvae.py` 的 `decode_temporal`。本地副本原来在 `D:\MinmaxH3\runs\official_vae\klvae.py`。
- `AutoencoderKLLegacy` 的结构配置：`Ref2VA/video_vae/source/config.json`。
- diffusers 管线使用的 VAE 配置：官方仓库 `vae/config.json`（`_class_name = AutoencoderKLMiniMaxH3`，同样有 `clip_length: 17` 和 `token_drop: 3`）。这里是 `official/vae_config.json`，从 `main` 的 raw 文件下载。

## 工作流

`workflows/h3_watch_stock_api.json` 是生成 `D:\MinmaxH3\ComfyUI\output\video\h3_watch_stock.mp4` 时提交给 ComfyUI `/prompt` 的 API 图。来源脚本：`D:\MinmaxH3\runs\h3_watch_queue.py`。

`workflows/h3_cmp_22frame_pruned_api.json` 和 `workflows/h3_cmp_22frame_full_api.json` 是 22 帧对照。来源脚本：`D:\MinmaxH3\runs\h3_cmp_queue.py`。两者只有 `unet_name` 和 `filename_prefix` 不同。
