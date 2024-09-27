import gc
import torch
import numpy as np
import pandas as pd
import open_clip
from transformers import (
    FlavaProcessor,
    FlavaForPreTraining,
    BertTokenizer,
    FlavaFeatureExtractor,
)
from transformers import AltCLIPModel, AltCLIPProcessor
from transformers import AlignProcessor, AlignModel, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = {
    "openai": [
        # ["ViT-B-32", "negCLIP.pt"], first download and add negClip weights in this directory
        ["EVA01-g-14", "laion400m_s11b_b41k"],
        ["EVA02-L-14", "merged2b_s4b_b131k"],
        ["RN50x64", "openai"],
        ["ViT-B-16", "openai"],
        ["ViT-B-32", "openai"],
        ["ViT-L-14", "openai"],
        ["coca_ViT-B-32", "laion2b_s13b_b90k"],
        ["coca_ViT-L-14", "laion2b_s13b_b90k"],
    ],
    "align": ["kakaobrain/align-base"],
    "alt": ["BAAI/AltCLIP"],
    "flava": ["facebook/flava-full"],
}


def preprocess_activity(activity):
    return activity.replace("_", " ").lower()


def reverse_gender(gender):
    return "man" if gender == "woman" else "woman"


def load_open_clip(activities, report_dict):
    for base_name, pretrained in CONFIG["openai"]:
        model, _, preprocess = open_clip.create_model_and_transforms(
            base_name, pretrained=pretrained
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(base_name)
        for activity in activities:
            tokenized_text = tokenizer(
                f"A person is {preprocess_activity(activity)}"
            ).to(device)
            tokenized_male_text = tokenizer(
                f"A man is {preprocess_activity(activity)}"
            ).to(device)
            tokenized_female_text = tokenizer(
                f"A woman is {preprocess_activity(activity)}"
            ).to(device)
            with torch.no_grad():
                text_features = model.encode_text(tokenized_text)
                text_features_norm = text_features.norm(dim=-1)
                male_text_features = model.encode_text(tokenized_male_text)
                male_text_features_norm = male_text_features.norm(dim=-1)
                female_text_features = model.encode_text(tokenized_female_text)
                female_text_features_norm = female_text_features.norm(dim=-1)
                male_sim = (
                    (text_features @ male_text_features.T)
                    / (text_features_norm * male_text_features_norm)
                ).item()
                female_sim = (
                    (text_features @ female_text_features.T)
                    / (text_features_norm * female_text_features_norm)
                ).item()
                sim_probs = torch.tensor([male_sim, female_sim]).softmax(dim=-1)
                male_sim_prob, female_sim_prob = (
                    sim_probs[0].item(),
                    sim_probs[1].item(),
                )
            report_dict["model"].append(f"{base_name} {pretrained}")
            report_dict["activity"].append(activity)
            report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
            report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()


def load_align(activities, report_dict):
    for model_name in CONFIG["align"]:
        model = AlignModel.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for activity in activities:
            tokenized_text = tokenizer(
                f"A person is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            tokenized_male_text = tokenizer(
                f"A man is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            tokenized_female_text = tokenizer(
                f"A woman is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_features = model.get_text_features(**tokenized_text)
                text_features_norm = text_features.norm(dim=-1)
                male_text_features = model.get_text_features(**tokenized_male_text)
                male_text_features_norm = male_text_features.norm(dim=-1)
                female_text_features = model.get_text_features(**tokenized_female_text)
                female_text_features_norm = female_text_features.norm(dim=-1)
                male_sim = (
                    (text_features @ male_text_features.T)
                    / (text_features_norm * male_text_features_norm)
                ).item()
                female_sim = (
                    (text_features @ female_text_features.T)
                    / (text_features_norm * female_text_features_norm)
                ).item()
                sim_probs = torch.tensor([male_sim, female_sim]).softmax(dim=-1)
                male_sim_prob, female_sim_prob = (
                    sim_probs[0].item(),
                    sim_probs[1].item(),
                )
            report_dict["model"].append(f"{model_name}")
            report_dict["activity"].append(activity)
            report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
            report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()


def load_alt(activities, report_dict):
    for model_name in CONFIG["alt"]:
        model = AltCLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for activity in activities:
            tokenized_text = tokenizer(
                f"A person is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            tokenized_male_text = tokenizer(
                f"A man is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            tokenized_female_text = tokenizer(
                f"A woman is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_features = model.get_text_features(**tokenized_text)
                text_features_norm = text_features.norm(dim=-1)
                male_text_features = model.get_text_features(**tokenized_male_text)
                male_text_features_norm = male_text_features.norm(dim=-1)
                female_text_features = model.get_text_features(**tokenized_female_text)
                female_text_features_norm = female_text_features.norm(dim=-1)
                male_sim = (
                    (text_features @ male_text_features.T)
                    / (text_features_norm * male_text_features_norm)
                ).item()
                female_sim = (
                    (text_features @ female_text_features.T)
                    / (text_features_norm * female_text_features_norm)
                ).item()
                sim_probs = torch.tensor([male_sim, female_sim]).softmax(dim=-1)
                male_sim_prob, female_sim_prob = (
                    sim_probs[0].item(),
                    sim_probs[1].item(),
                )
            report_dict["model"].append(f"{model_name}")
            report_dict["activity"].append(activity)
            report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
            report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()


def load_flava(activities, report_dict):
    for model_name in CONFIG["flava"]:
        model = FlavaForPreTraining.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(model_name)
        for activity in activities:
            tokenized_text = tokenizer(
                f"A person is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            tokenized_male_text = tokenizer(
                f"A man is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            tokenized_female_text = tokenizer(
                f"A woman is {preprocess_activity(activity)}",
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_features = model.flava.get_text_features(**tokenized_text)[:, 0, :]
                text_features_norm = text_features.norm(dim=-1)
                male_text_features = model.flava.get_text_features(
                    **tokenized_male_text
                )[:, 0, :]
                male_text_features_norm = male_text_features.norm(dim=-1)
                female_text_features = model.flava.get_text_features(
                    **tokenized_female_text
                )[:, 0, :]
                female_text_features_norm = female_text_features.norm(dim=-1)
                male_sim = (
                    (text_features @ male_text_features.T)
                    / (text_features_norm * male_text_features_norm)
                ).item()
                female_sim = (
                    (text_features @ female_text_features.T)
                    / (text_features_norm * female_text_features_norm)
                ).item()
                sim_probs = torch.tensor([male_sim, female_sim]).softmax(dim=-1)
                male_sim_prob, female_sim_prob = (
                    sim_probs[0].item(),
                    sim_probs[1].item(),
                )
            report_dict["model"].append(f"{model_name}")
            report_dict["activity"].append(activity)
            report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
            report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()
