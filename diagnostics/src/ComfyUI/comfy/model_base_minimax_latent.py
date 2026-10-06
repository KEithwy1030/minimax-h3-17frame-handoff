# Verbatim excerpts from D:\MinmaxH3\ComfyUI\comfy\model_base.py
# process_latent_in / process_latent_out are NOT methods of comfy.sd.VAE.
# BaseModel forwards them to latent_format.process_in / process_out.
# MiniMaxH3Video.scale_factor is 1.0 and that class does not override process_in,
# so the video latent is not rescaled here. MiniMaxH3 only rescales the audio slice.

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\model_base.py lines 370-381 -----
        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(self, unet_state_dict, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):

# ----- verbatim D:\MinmaxH3\ComfyUI\comfy\model_base.py lines 2136-2162 -----
class MiniMaxH3(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW_AV, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.minimax.model.MiniMaxH3Model)

    def audio_scale(self):
        """Scale the sampler carries the audio stream at, 1.0 when not sampling the packed latent."""
        if self.latent_shapes is None or len(self.latent_shapes) < 2:
            return 1.0
        return self.model_sampling.audio_scale

    def _scale_audio_slice(self, latent, scale):
        # the sampler carries the audio stream scaled onto the video schedule
        if scale == 1.0:
            return latent
        if latent.is_nested:  # the x0 output hands back the unpacked view
            streams = latent.unbind()
            return comfy.nested_tensor.NestedTensor([streams[0], streams[1] * scale] + list(streams[2:]))
        n = math.prod(self.latent_shapes[0][1:])
        latent = latent.clone()
        latent[..., n:] *= scale
        return latent

    def process_latent_in(self, latent):
        return self._scale_audio_slice(super().process_latent_in(latent), self.audio_scale())

    def process_latent_out(self, latent):
        return super().process_latent_out(self._scale_audio_slice(latent, 1.0 / self.audio_scale()))

