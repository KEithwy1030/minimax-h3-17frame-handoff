"""Experimental temporal decode. Does not modify ComfyUI's vae.py.

Each 17-frame window is decoded with the previous chunk's last
`context_tokens` latent tokens prepended. Frames that belong to those
warmup tokens are discarded. The frames kept for the current window use
the same pre-padding, overlap blend, and output length as
`MiniMaxH3VideoVAE.decode_temporal`.
"""
import torch


def decode_with_context(model, z, context_tokens=2):
    """z is already denormalized, the same tensor decode_temporal receives.

    Returns (pixels float32 in [0, 1] shaped like decode_temporal, plan list).
    """
    import comfy.model_management

    if context_tokens < 1:
        raise ValueError("context_tokens must be >= 1")

    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    dec = torch.empty(
        model.decode_output_shape(z.shape),
        dtype=torch.float32,
        device=comfy.model_management.intermediate_device(),
    )
    pad_tokens, num_chunks = model._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)

    dec_overlap = None
    write_pos = 0
    plan = []

    def write_part(part):
        nonlocal write_pos
        part_frames = int(part.shape[2])
        if part_frames <= 0:
            return 0, 0
        part = model._finalize_pixels(part)
        start = write_pos
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos : write_pos + copy_frames].copy_(part[:, :, :copy_frames])
            write_pos += copy_frames
        del part
        return start, copy_frames

    for i in range(num_chunks):
        t_start = i * model.tokens_chunk_size
        t_end = t_start + model.tokens_chunk_size + model.token_overlap
        ctx = context_tokens if i > 0 else 0
        t_ctx = max(0, t_start - ctx)
        clip_z = z[:, :, t_ctx:t_end]
        clip_dec = model._adaptive_decode(clip_z)
        warmup_frames = (t_start - t_ctx) * model.vae_ratio_t
        if warmup_frames:
            clip_dec = clip_dec[:, :, warmup_frames:]

        dropped_local = []
        output_spans = []
        for j in range(split_count):
            f_start = j * chunk_dec
            f_end = min(f_start + chunk_dec, clip_dec.shape[2])
            drop_end = min(f_start + model.frame_pre_padding, f_end)
            dropped_local.extend(range(f_start, drop_end))
            clip_chunk = clip_dec[:, :, f_start:f_end]
            clip_chunk = clip_chunk[:, :, model.frame_pre_padding :]
            if j == 0:
                if dec_overlap is not None:
                    clip_chunk = model.blend(
                        dec_overlap, clip_chunk, model.frame_overlap, dim=-3
                    )
                    dec_overlap = None
                start, copied = write_part(clip_chunk)
                output_spans.append([start, start + copied])
            else:
                dec_overlap = clip_chunk.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            start, copied = write_part(dec_overlap)
            output_spans.append([start, start + copied])
            dec_overlap = None

        plan.append(
            {
                "chunk": i,
                "context_latent_t": [t_ctx, t_start],
                "current_latent_t": [t_start, min(t_end, z.shape[2])],
                "tokens_into_decoder": int(clip_z.shape[2]),
                "warmup_frames_discarded": int(warmup_frames),
                "dropped_local_frames_after_warmup_removed": dropped_local,
                "output_frame_spans": output_spans,
            }
        )
        del clip_dec, clip_z
        if z.is_cuda:
            torch.cuda.empty_cache()

    return dec, plan
