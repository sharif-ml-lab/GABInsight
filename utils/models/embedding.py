import torch
from torch import nn
from transformers import AutoImageProcessor, Swinv2Model, ViTImageProcessor, ViTModel
from transformers import logging as transformers_logging


transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class Embedding(nn.Module):
    def __init__(self, device) -> None:
        super(Embedding, self).__init__()
        self.device = device

    def forward(self, image):
        pass


class SwinV2Tiny(Embedding):
    def __init__(
        self,
        device,
        model_name="microsoft/swinv2-tiny-patch4-window8-256",
    ):
        super(SwinV2Tiny, self).__init__(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Swinv2Model.from_pretrained(model_name).to(self.device)

    def forward(self, image):
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state
        return embedding.reshape(1, -1)


class ViTLarge(Embedding):
    def __init__(
        self,
        device,
        model_name="google/vit-base-patch16-224",
    ):
        super(ViTLarge, self).__init__(device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def forward(self, image):
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state
        return embedding.reshape(1, -1)
