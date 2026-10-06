"""Index plan shared by ComfyUI and official decode_temporal when isolated frames are off.

This repeats the index arithmetic in both decode_temporal implementations.
The GPU runner checks it against the token count actually passed to _adaptive_decode.
"""
import math


def temporal_constants(clip_length=17, token_drop=3, vae_ratio_t=4):
    tokens_chunk_size = math.ceil(clip_length / vae_ratio_t)
    token_overlap = (-token_drop) % tokens_chunk_size
    frame_pre_padding = (-clip_length) % vae_ratio_t
    frame_overlap = max(token_overlap * vae_ratio_t - frame_pre_padding, 0)
    return {
        "clip_length": clip_length,
        "token_drop": token_drop,
        "vae_ratio_t": vae_ratio_t,
        "tokens_chunk_size": tokens_chunk_size,
        "token_overlap": token_overlap,
        "frame_pre_padding": frame_pre_padding,
        "frame_overlap": frame_overlap,
        "chunk_dec": tokens_chunk_size * vae_ratio_t,
        "split_count": int(token_drop > 0) + 1,
    }


def comfy_chunks(z_len, constants=None):
    c = constants or temporal_constants()
    pseudo = z_len + c["token_drop"]
    pad_tokens = (-pseudo) % c["tokens_chunk_size"]
    pseudo += pad_tokens
    num_chunks = pseudo // c["tokens_chunk_size"] - int(c["token_drop"] > 0)
    if num_chunks < 1:
        pad_tokens += c["tokens_chunk_size"]
        num_chunks += 1
    return pad_tokens, num_chunks


def official_chunks(z_len, constants=None, isolated_token_num=0):
    c = constants or temporal_constants()
    pseudo = z_len - isolated_token_num + c["token_drop"]
    pad_tokens = 0
    remainder = pseudo % c["tokens_chunk_size"]
    if remainder != 0:
        pad_tokens = c["tokens_chunk_size"] - remainder
        pseudo += pad_tokens
    num_chunks = pseudo // c["tokens_chunk_size"] - int(c["token_drop"] > 0)
    return pad_tokens, num_chunks


def chunk_rows(z_len, pad_tokens, num_chunks, constants=None):
    c = constants or temporal_constants()
    total = z_len + pad_tokens
    rows = []
    out = 0
    for i in range(num_chunks):
        t0 = i * c["tokens_chunk_size"]
        t1 = t0 + c["tokens_chunk_size"] + c["token_overlap"]
        a0 = min(t0, total)
        a1 = min(t1, total)
        tokens = a1 - a0
        frames = tokens * c["vae_ratio_t"]
        dropped = []
        kept = []
        for j in range(c["split_count"]):
            fs = j * c["chunk_dec"]
            fe = min(fs + c["chunk_dec"], frames)
            drop_end = min(fs + c["frame_pre_padding"], fe)
            dropped.extend(range(fs, drop_end))
            kept.append(list(range(drop_end, fe)))
        written = []
        if kept:
            n0 = len(kept[0])
            written.append([out, out + n0])
            out += n0
        if i == num_chunks - 1 and len(kept) > 1:
            n1 = len(kept[1])
            written.append([out, out + n1])
            out += n1
        rows.append(
            {
                "chunk": i,
                "latent_t": [a0, a1],
                "tokens_into_decoder": tokens,
                "decoded_frames": frames,
                "dropped_local_frames": dropped,
                "kept_local_frames_j0": kept[0] if kept else [],
                "kept_local_frames_j1_overlap": kept[1] if len(kept) > 1 else [],
                "output_frame_spans": written,
            }
        )
    return rows, out
