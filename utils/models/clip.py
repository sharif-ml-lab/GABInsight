import torch
from torch import nn
import open_clip
from transformers import AlignProcessor, AlignModel
from transformers import AltCLIPModel, AltCLIPProcessor
from transformers import FlavaProcessor, FlavaModel
from transformers import logging as transformers_logging
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from transformers import AutoProcessor, BlipModel
from transformers import (
    FlavaProcessor,
    FlavaForPreTraining,
    BertTokenizer,
    FlavaFeatureExtractor,
)
import torch.nn.functional as F
import torch


class Clip(nn.Module):
    def __init__(self, device) -> None:
        super(Clip, self).__init__()
        self.device = device

    def forward(self, image):
        pass


class Vilt(Clip):
    def __init__(self, device, model_name="dandelin/vilt-b32-finetuned-coco"):
        super(Vilt, self).__init__(device)
        self.model = ViltForImageAndTextRetrieval.from_pretrained(model_name).to(device)
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.device = device

    def forward(self, image, text_list):
        with torch.no_grad():
            scores = dict()
            for text in text_list:
                self.model.eval()
                encoding = self.processor(image, text, return_tensors="pt").to(
                    self.device
                )
                outputs = self.model(**encoding)
                scores[text] = outputs.logits[0, :].item()
            output = list(F.softmax(torch.tensor(list(scores.values()))).numpy())
        return [output]


class Flava(Clip):
    def __init__(self, device, model_name="facebook/flava-full"):
        super(Flava, self).__init__(device)

        self.model = FlavaForPreTraining.from_pretrained(model_name).eval().to(device)
        self.feature_extractor = FlavaFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.processor = FlavaProcessor.from_pretrained(model_name)
        self.device = device

    def forward(self, image, text_list):
        with torch.no_grad():
            text_input = self.tokenizer(
                text=text_list, return_tensors="pt", padding="max_length", max_length=77
            ).to(self.device)
            self.model.eval()
            text_feats = (
                self.model.flava.get_text_features(**text_input).cpu().numpy()[:, 0, :]
            )
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(
                self.device
            )
            image_feats = (
                self.model.flava.get_image_features(**inputs).cpu().numpy()[:, 0, :]
            )
            scores = image_feats @ text_feats.T
            prob = torch.tensor(scores).softmax(dim=1).cpu().numpy()
        return prob


class Blip(Clip):
    def __init__(self, device, model_name="Salesforce/blip-image-captioning-base"):
        super(Blip, self).__init__(device)
        self.model = BlipModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(self, image, text_list):
        with torch.no_grad():
            inputs = self.processor(
                text=text_list, images=image, return_tensors="pt", padding=True
            ).to(self.device)
            self.model.eval()
            outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).cpu().numpy()


class AltClip(Clip):
    def __init__(self, device, model_name="BAAI/AltCLIP"):
        super(AltClip, self).__init__(device)
        self.model = AltCLIPModel.from_pretrained(model_name).to(device)
        self.processor = AltCLIPProcessor.from_pretrained(model_name)

    def forward(self, image, text_list):
        with torch.no_grad():
            inputs = self.processor(
                text=text_list, images=image, return_tensors="pt", padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).cpu().numpy()


class AlignClip(Clip):
    def __init__(self, device, model_name="kakaobrain/align-base"):
        super(AlignClip, self).__init__(device)
        self.model = AlignModel.from_pretrained(model_name).to(device)
        self.processor = AlignProcessor.from_pretrained(model_name)

    def forward(self, image, text_list):
        with torch.no_grad():
            inputs = self.processor(
                text=text_list, images=image, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).cpu().numpy()


class ViTOpenAIClip(Clip):
    def __init__(
        self,
        device,
        base_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    ):
        super(ViTOpenAIClip, self).__init__(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=base_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(base_name)

    def forward(self, image, text_list):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(text_list).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs.cpu().numpy()
