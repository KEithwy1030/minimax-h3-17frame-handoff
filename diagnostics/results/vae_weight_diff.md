# 本地 fp16 VAE 与官方权重的差异

A = `D:\MinmaxH3\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors`，5207808496 字节。这是 ComfyUI 一直在用的本地视频 VAE。
B = `D:\MinmaxH3\runs\official_fp32_vae\model.safetensors`，10415548320 字节。来自 Hugging Face `MiniMaxAI/MiniMax-H3` 的 `Ref2VA/video_vae/source/model.safetensors`。比较前先把 B 的每个张量转成 fp16。

相对 L2 = `||A-B||_2 / ||B||_2`，B 已是 fp16。最大绝对差是逐元素 `|A-B|` 的最大值。排序按相对 L2 从大到小。

A 有 562 个键，B 有 560 个键。共有键 560 个，其中数值完全相同 557 个，有差异 3 个。
A 的 safetensors metadata 有一个键 `minimax_h3_video_vae`，内容是 JSON，里面的 `latents_mean` / `latents_std` 与 `vae.py` 的字面量一致。B 没有 metadata。

## 只在一边出现的键

| 类别 | 键 | shape | dtype | 与 vae.py 字面量的最大绝对差 |
| --- | --- | --- | --- | --- |
| only_A | `latents_mean` | (24,) | float16 | 0.00040888786 |
| only_A | `latents_std` | (24,) | float16 | 0.00083565712 |

这两个张量只存在于 A。和 `vae.py` 里写死的 24 个数相比，最大绝对差分别是 4.1e-4 和 8.4e-4，对应 fp16 保存时的舍入，不是另一套统计量。

## shape 不同、含 NaN/Inf、或整张量全零

| 类别 | 键 | 说明 |
| --- | --- | --- |
| nan_inf_or_zero | `decoder.mask_token` | nan_or_inf A=False B=False; all_zero A=True B=True; decoder token |

## 差异最大的共有键

有差异的共有键只有 3 个，不到 30 个。其余 557 个共有键的最大绝对差是 0。

| 键 | shape | 最大绝对差 | 相对 L2 | 有差异的元素 |
| --- | --- | --- | --- | --- |
| `decoder.transformer_blocks.0.attn.to_out.bias` | (2048,) | 0.0906448 | 6.11081 | 2048/2048 |
| `decoder.register_tokens` | (1, 4, 2048) | 0.114838 | 1.86995 | 8192/8192 |
| `decoder.transformer_blocks.0.attn.to_out.weight` | (2048, 2048) | 0.164291 | 1.36886 | 2044281/4194304 |

## 解码相关张量

decoder 下没有卷积权重。因果卷积在 encoder 里，键名带 `conv`。
decoder 的 norm 是各层 `norm1` / `norm2` 和 `norm_out`。`post_quant_conv`、`decoder.proj_out`、`decoder.mask_token`、`decoder.register_tokens` 单独列出。

encoder 的因果卷积、各层 norm、`post_quant_conv`、`decoder.proj_out` 都在那 557 个完全相同的键里。

重点张量里有差异的共 1 个。另外两个有差异的键是 decoder 第 0 层 attention 的 `to_out`，不在卷积或 norm 里，已列在上一张表。

换成由 B 转出的 `minimax_h3_video_vae_official_fp16.safetensors` 后，22 帧那次第 16→17 帧下降 0.66%，124 帧各接缝下降在 0.50% 到 0.80% 之间。

| 标记 | 键 | 最大绝对差 | 相对 L2 |
| --- | --- | --- | --- |
| decoder token | `decoder.register_tokens` | 0.114838 | 1.86995 |
