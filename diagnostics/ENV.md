# 环境

测量实验 2–5 时的环境。实验 1 只读 latent，不用 GPU。

- ComfyUI：`D:\MinmaxH3\ComfyUI`，commit `8a33128f2f8c5585c57486c07de481241e70a39c`
- `comfy/ldm/minimax/vae.py` 相对这个 commit 没有改动
- `comfy/sd.py` 工作区比这个 commit 多 4 行：MiniMax H3 video VAE 分支里 `disable_offload = True`
- Python：3.13.13，`D:\MinmaxH3\ComfyUI\python_embeded\python.exe`
- PyTorch：2.11.0+cu130
- CUDA：13.0
- GPU：NVIDIA GeForce RTX 3060，12288 MiB

这次实际拿来解码和往返的权重：

- 文件：`D:\MinmaxH3\runs\official_fp32_vae\model.safetensors`
- 来源：`https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/Ref2VA/video_vae/source/model.safetensors`
- 大小：10415548320 字节
- 计算 dtype：float16。文件本身是 fp32，12GB 显存放不下，所以读入时逐张量转成 fp16
- 官方模型 `load_state_dict`：missing 0，unexpected 0
- `clip_length = 17`，`token_drop = 3`，`decoder_tiling = True`，tile 256，overlap 64
- 这个文件没有 `latents_mean` / `latents_std`。ComfyUI 加载时警告缺这两项，缓冲区保留 `vae.py` 里写死的数。反归一化用的也是这组数

本地还有 `D:\MinmaxH3\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors`，5207808496 字节。实验 2–5 没有用它。

官方解码时出现一次 PyTorch 警告：RMSNorm 的输入是 float、权重是 float16。成片亮度仍然和 ComfyUI 解码差在 0.02 以内。
