# Verbatim excerpts from D:\MinmaxH3\ComfyUI\comfy\sd.py
# ComfyUI commit 8a33128f2f8c5585c57486c07de481241e70a39c
# Working tree adds 4 lines inside the MiniMax H3 video VAE branch:
#   self.disable_offload = True
#   plus the three comment lines immediately above it.
# The VAE class has no process_latent_in / process_latent_out methods.
# Those methods are on the diffusion model; see model_base_minimax_latent.py.
# MiniMax H3 latent denormalization (latents_mean / latents_std) is inside
# MiniMaxH3VideoVAE.decode in comfy/ldm/minimax/vae.py, not in this VAE wrapper.
# This wrapper sets process_output = identity for the H3 video VAE.

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\sd.py lines 914-927 -----
            elif "decoder.22.bias" in sd: # taehv, taew and lighttae
                self.latent_channels = sd["decoder.1.weight"].shape[1]
                self.latent_dim = 3
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                self.upscale_index_formula = (4, 16, 16)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                self.downscale_index_formula = (4, 16, 16)
                if self.latent_channels == 24 and sd["decoder.22.bias"].shape[0] == 12: # MiniMax H3
                    self.first_stage_model = comfy.taesd.taehv.TAEHV(latent_channels=self.latent_channels, latent_format=None)
                    self.process_input = self.process_output = lambda image: image
                    self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
                    self.downscale_ratio = (lambda a: max(1, (a - 1) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
                    self.memory_used_encode = lambda shape, dtype: (400 * ((shape[-3] + 16) // 17) * shape[-2] * shape[-1] * model_management.dtype_size(dtype))
                    self.memory_used_decode = lambda shape, dtype: ((260 * 16 * 16 + shape[1] * shape[-3]) * shape[-2] * shape[-1] * model_management.dtype_size(dtype))

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\sd.py lines 995-1054 -----
            elif "decoder.transformer_blocks.0.scale1" in sd and "encoder.down.5.block.0.conv1.weight" in sd:  # MiniMax H3 video VAE
                minimax_ops = comfy.ops.disable_weight_init
                minimax_quant = comfy.utils.detect_layer_quantization(sd, "")
                if minimax_quant is not None:  # int8+convrot quantized decoder
                    minimax_ops = comfy.ops.mixed_precision_ops(minimax_quant, dtype if dtype is not None else torch.float16)
                self.first_stage_model = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE(operations=minimax_ops)
                self.latent_channels = 24
                self.latent_dim = 3
                # frames 17k+5 <-> latents 5k+2, 16x spatial
                self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
                self.upscale_index_formula = (4, 16, 16)
                self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
                self.downscale_index_formula = (4, 16, 16)
                self.working_dtypes = [torch.float16, torch.float32]
                # the model tiles internally (256px spatial, 17-frame temporal chunks)
                self.handles_tiling = True
                # H3 VAE has non-parameter buffers in its ViT decoder. Partial
                # offload can leave them on CPU while activations are on CUDA.
                # Keep this VAE together on its selected device for correctness.
                self.disable_offload = True
                # decode finalizes straight to [0, 1] while streaming chunks out
                self.process_output = lambda image: image
                # one decoded temporal chunk (with overlap) is all that ever sits in VRAM
                chunk_frames = (self.first_stage_model.tokens_chunk_size + self.first_stage_model.token_overlap) * self.first_stage_model.vae_ratio_t

                def estimate_encode_memory(frames, height, width, dtype):
                    fixed = 110_000_000 if frames == 1 else 1_300_000_000
                    elements_per_pixel = 7 if frames == 1 else 9.5
                    # only one clip of the input video is ever resident on the GPU
                    frames = min(frames, self.first_stage_model.clip_length)
                    return (elements_per_pixel * frames * height * width + fixed) * model_management.dtype_size(dtype) * 1.03

                def estimate_decode_memory(frames, height, width, dtype):
                    fixed = 110_000_000 if frames <= 22 else 270_000_000
                    frames = min(frames, chunk_frames + 2)
                    return (9.5 * frames * height * width + fixed) * model_management.dtype_size(dtype) * 1.03

                self.memory_used_encode = lambda shape, dtype: estimate_encode_memory(shape[2], shape[3], shape[4], dtype)
                self.memory_used_decode = lambda shape, dtype: estimate_decode_memory(self.upscale_ratio[0](shape[2]), shape[3] * self.upscale_ratio[1], shape[4] * self.upscale_ratio[2], dtype)
            elif "pre_block.attn.zero_k_bias" in sd:  # MiniMax H3 audio VAE (DAC encoder + BigVGAN decoder)
                self.first_stage_model = comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE()
                self.latent_channels = 32
                self.output_channels = 2
                self.pad_channel_value = "replicate"
                self.audio_sample_rate = 32000
                self.upscale_ratio = 800
                self.downscale_ratio = 800
                self.latent_dim = 2  # [B, 32, stereo 2, T]
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float32]
                # encode gets the waveform shape [B, 2, samples], decode the latent shape [B, 32, 2, T]
                def estimate_encode_memory(samples, dtype):
                    return (900 * samples + 105_000_000) * model_management.dtype_size(dtype) * 1.03

                def estimate_decode_memory(samples, dtype):
                    return max(42_000_000, 220 * samples + 20_000_000) * model_management.dtype_size(dtype) * 1.03

                self.memory_used_encode = lambda shape, dtype: estimate_encode_memory(shape[2], dtype)
                self.memory_used_decode = lambda shape, dtype: estimate_decode_memory(shape[-1] * self.upscale_ratio, dtype)

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\sd.py lines 1224-1300 -----
    def decode(self, samples_in, vae_options={}):
        self.throw_exception_if_invalid()
        pixel_samples = None
        do_tile = False
        if self.latent_dim == 2 and samples_in.ndim == 5:
            samples_in = samples_in[:, :, 0]

        with model_management.cuda_device_context(self.device):
            try:
                memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
                model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
                free_memory = self.patcher.get_free_memory(self.device)
                batch_number = int(free_memory / memory_used)
                batch_number = max(1, batch_number)

                # Pre-allocate output for VAEs that support direct buffer writes
                preallocated = False
                if getattr(self.first_stage_model, 'comfy_has_chunked_io', False):
                    pixel_samples = torch.empty(self.first_stage_model.decode_output_shape(samples_in.shape), device=self.output_device, dtype=self.vae_output_dtype())
                    preallocated = True

                for x in range(0, samples_in.shape[0], batch_number):
                    samples = samples_in[x:x + batch_number].to(device=self.device, dtype=self.vae_dtype)
                    if preallocated:
                        self.first_stage_model.decode(samples, output_buffer=pixel_samples[x:x+batch_number], **vae_options)
                    else:
                        out = self.first_stage_model.decode(samples, **vae_options).to(device=self.output_device, dtype=self.vae_output_dtype(), copy=True)
                        if pixel_samples is None:
                            pixel_samples = torch.empty((samples_in.shape[0],) + tuple(out.shape[1:]), device=self.output_device, dtype=self.vae_output_dtype())
                        pixel_samples[x:x+batch_number].copy_(out)
                        del out
                    self.process_output(pixel_samples[x:x+batch_number])
            except Exception as e:
                model_management.raise_non_oom(e)
                logging.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                #NOTE: We don't know what tensors were allocated to stack variables at the time of the
                #exception and the exception itself refs them all until we get out of this except block.
                #So we just set a flag for tiler fallback so that tensor gc can happen once the
                #exception is fully off the books.
                do_tile = True

            if do_tile:
                pixel_samples = None
                comfy.model_management.soft_empty_cache()
                dims = samples_in.ndim - 2
                if dims == 1 or self.extra_1d_channel is not None:
                    pixel_samples = self.decode_tiled_1d(samples_in)
                elif dims == 2:
                    if self.handles_tiling:
                        tile = 256 // self.spacial_compression_decode()
                        overlap = tile // 4
                        pixel_samples = self._decode_tiled_owned(samples_in, tile_x=tile, tile_y=tile, overlap=overlap)
                    else:
                        pixel_samples = self.decode_tiled_(samples_in)
                elif dims == 3:
                    tile = 256 // self.spacial_compression_decode()
                    overlap = tile // 4
                    if self.handles_tiling:
                        memory_used = self.memory_used_decode(self._tile_bounded_shape(samples_in.shape, tile, tile, None), self.vae_dtype)
                        model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
                        pixel_samples = self._decode_tiled_owned(samples_in, tile_x=tile, tile_y=tile, overlap=overlap)
                    else:
                        # Reserve as much as an untiled decode could use (capped by what the device can provide), then size the tiles to fill that reservation:
                        # shrink the temporal tile until one tile fits, then grow the spatial tile while it still fits.
                        budget = min(memory_used, int(model_management.get_total_memory(self.device) * 0.8))
                        model_management.load_models_gpu([self.patcher], memory_required=budget, force_full_load=self.disable_offload)
                        tile_t = samples_in.shape[2]
                        est = lambda tt, txy: self.memory_used_decode(self._tile_bounded_shape(samples_in.shape, txy, txy, tt), self.vae_dtype)
                        while tile_t > 2 and est(tile_t, tile) > budget:
                            tile_t = -(-tile_t // 2)
                        while tile * 2 <= max(samples_in.shape[3], samples_in.shape[4]) and est(tile_t, tile * 2) <= budget:
                            tile *= 2
                        overlap = tile // 4
                        pixel_samples = self.decode_tiled_3d(samples_in, tile_t=tile_t, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))

        pixel_samples = pixel_samples.to(self.output_device).movedim(1,-1)
        return pixel_samples

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\sd.py lines 1362-1418 -----
    def encode(self, pixel_samples):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        do_tile = False
        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            if not self.not_video:
                pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
            else:
                pixel_samples = pixel_samples.unsqueeze(2)

        with model_management.cuda_device_context(self.device):
            try:
                memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
                model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
                free_memory = self.patcher.get_free_memory(self.device)
                batch_number = int(free_memory / max(1, memory_used))
                batch_number = max(1, batch_number)
                samples = None
                for x in range(0, pixel_samples.shape[0], batch_number):
                    pixels_in = self.process_input(pixel_samples[x:x + batch_number]).to(self.vae_dtype)
                    if getattr(self.first_stage_model, 'comfy_has_chunked_io', False):
                        out = self.first_stage_model.encode(pixels_in, device=self.device)
                    else:
                        pixels_in = pixels_in.to(self.device)
                        out = self.first_stage_model.encode(pixels_in)
                    out = out.to(self.output_device).to(dtype=self.vae_output_dtype())
                    if samples is None:
                        samples = torch.empty((pixel_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device, dtype=self.vae_output_dtype())
                    samples[x:x + batch_number] = out

            except Exception as e:
                model_management.raise_non_oom(e)
                logging.warning("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
                #NOTE: We don't know what tensors were allocated to stack variables at the time of the
                #exception and the exception itself refs them all until we get out of this except block.
                #So we just set a flag for tiler fallback so that tensor gc can happen once the
                #exception is fully off the books.
                do_tile = True

            if do_tile:
                comfy.model_management.soft_empty_cache()
                if self.latent_dim == 3:
                    tile = 256
                    overlap = tile // 4
                    if self.handles_tiling:
                        samples = self._encode_tiled_owned(pixel_samples, tile_x=tile, tile_y=tile, overlap=overlap)
                    else:
                        samples = self.encode_tiled_3d(pixel_samples, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))
                elif self.latent_dim == 1 or self.extra_1d_channel is not None:
                    samples = self.encode_tiled_1d(pixel_samples)
                else:
                    samples = self.encode_tiled_(pixel_samples)

        if self.format_encoded is not None:
            samples = self.format_encoded(samples)
        return samples

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\sd.py lines 1475-1497 -----
    def spacial_compression_decode(self):
        try:
            return self.upscale_ratio[-1]
        except:
            return self.upscale_ratio

    def spacial_compression_encode(self):
        try:
            return self.downscale_ratio[-1]
        except:
            return self.downscale_ratio

    def temporal_compression_decode(self):
        try:
            return round(self.upscale_ratio[0](8192) / 8192)
        except:
            return None

    def is_dynamic(self):
        # A VAE built from a state dict with no detectable VAE weights returns early
        # from __init__ ("No VAE weights detected") before self.patcher is assigned.
        patcher = getattr(self, "patcher", None)
        return patcher is not None and patcher.is_dynamic()

