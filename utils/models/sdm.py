import torch
from torch import nn
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, AutoencoderKL
from diffusers import logging as diffusers_logging


diffusers_logging.set_verbosity_error()
diffusers_logging.disable_progress_bar()


class SDM(nn.Module):
    def __init__(self, device, half_precision=False) -> None:
        super(SDM, self).__init__()
        self.device = device

    def forward(self, sentence):
        pass


class XlargeTurbuSDM(SDM):
    def __init__(self, device, model_name="stabilityai/sdxl-turbo"):
        super(XlargeTurbuSDM, self).__init__(device, False)
        self.model = AutoPipelineForText2Image.from_pretrained(model_name).to(
            self.device
        )

    def forward(self, prompt):
        image = self.model(
            prompt=prompt, guidance_scale=0.0, num_inference_steps=1
        ).images[0]
        return image


class XlargeVAESDM(SDM):
    def __init__(
        self,
        device,
        vae_name="stabilityai/sd-vae-ft-mse",
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        refiner_name="stabilityai/stable-diffusion-xl-refiner-1.0",
    ):
        diffusers_logging.set_verbosity_error()
        super(XlargeVAESDM, self).__init__(device, False)
        vae = AutoencoderKL.from_pretrained(vae_name).to(self.device)
        self.model = DiffusionPipeline.from_pretrained(
            model_name,
            vae=vae,
        ).to(self.device)
        self.refiner = DiffusionPipeline.from_pretrained(refiner_name).to(self.device)

    def forward(self, prompt):
        n_steps = 40
        high_noise_frac = 0.73
        image = self.model(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
        return image
