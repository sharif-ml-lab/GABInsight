import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging


transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class SentenceEncoder(nn.Module):
    def __init__(self, device, half_precision=False) -> None:
        super(SentenceEncoder, self).__init__()
        self.device = device
        self.dtype = torch.float16 if half_precision else torch.float32

    def forward(self, sentence):
        pass


class MiniLMEncoder(SentenceEncoder):
    def __init__(self, device, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(MiniLMEncoder, self).__init__(device, False)
        self.model = SentenceTransformer(model_name).to(self.device)

    def forward(self, sentence):
        return self.model.encode(sentence, convert_to_tensor=True)


class MPNetEncoder(SentenceEncoder):
    def __init__(self, device, model_name="sentence-transformers/all-mpnet-base-v2"):
        super(MPNetEncoder, self).__init__(device, False)
        self.model = SentenceTransformer(model_name).to(self.device)

    def forward(self, sentence):
        return self.model.encode(sentence, convert_to_tensor=True)
