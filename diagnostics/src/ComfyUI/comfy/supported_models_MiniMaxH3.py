# Verbatim excerpt from D:\MinmaxH3\ComfyUI\comfy\supported_models.py
# latent_format = latent_formats.MiniMaxH3AV

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\supported_models.py lines 960-986 -----
class MiniMaxH3(supported_models_base.BASE):
    unet_config = {
        "image_model": "minimax_h3",
    }

    sampling_settings = {
        "shift": 12.0,
        "audio_shift": 3.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.MiniMaxH3AV

    memory_usage_factor = 0.114

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.MiniMaxH3(self, device=device)

    def clip_target(self, state_dict={}, prefix=""):
        pref = self.text_encoder_key_prefix[0]
        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_32b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.minimax.MiniMaxH3Tokenizer, comfy.text_encoders.minimax.te(**detect))

