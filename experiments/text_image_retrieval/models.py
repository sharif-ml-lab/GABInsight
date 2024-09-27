import gc
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
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


def load_open_clip(activities, report_dict, dataset_base_address, mode, gender):
    for base_name, pretrained in CONFIG["openai"]:
        print(base_name, pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            base_name, pretrained=pretrained
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(base_name)
        for activity in activities:
            if mode == "single":
                text = f"A {gender} is {preprocess_activity(activity)}"
            else:
                text = f"A {gender} is {preprocess_activity(activity)} and a {reverse_gender(gender)} is in the scene"
            tokenized_text = tokenizer(text).to(device)
            male_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/"
                )
                if name[0] != "."
            ]
            female_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/"
                )
                if name[0] != "."
            ]
            for i in range(np.min([len(male_images_names), len(female_images_names)])):
                male_image = (
                    preprocess(
                        Image.open(
                            f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/{male_images_names[i]}"
                        )
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                female_image = (
                    preprocess(
                        Image.open(
                            f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/{female_images_names[i]}"
                        )
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    text_features = model.encode_text(tokenized_text)
                    text_features_norm = text_features.norm(dim=-1)
                    male_image_features = model.encode_image(male_image)
                    male_image_features_norm = male_image_features.norm(dim=-1)
                    female_image_features = model.encode_image(female_image)
                    female_image_features_norm = female_image_features.norm(dim=-1)
                    male_sim = (
                        (text_features @ male_image_features.T)
                        / (text_features_norm * male_image_features_norm)
                    ).item()
                    female_sim = (
                        (text_features @ female_image_features.T)
                        / (text_features_norm * female_image_features_norm)
                    ).item()
                    sim_probs = torch.tensor([male_sim, female_sim]).softmax(dim=-1)
                    male_sim_prob, female_sim_prob = (
                        sim_probs[0].item(),
                        sim_probs[1].item(),
                    )
                report_dict["model"].append(f"{base_name} {pretrained}")
                report_dict["activity"].append(activity)
                report_dict["text"].append(text)
                report_dict["male_image_name"].append(male_images_names[i])
                report_dict["female_image_name"].append(female_images_names[i])
                report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
                report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()


def load_align(activities, report_dict, dataset_base_address, mode, gender):
    for model_name in CONFIG["align"]:
        model = AlignModel.from_pretrained(model_name).to(device)
        model.eval()
        processor = AlignProcessor.from_pretrained(model_name)
        for activity in activities:
            if mode == "single":
                text = f"A {gender} is {preprocess_activity(activity)}"
            else:
                text = f"A {gender} is {preprocess_activity(activity)} and a {reverse_gender(gender)} is in the scene"
            male_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/"
                )
                if name[0] != "."
            ]
            female_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/"
                )
                if name[0] != "."
            ]
            for i in range(np.min([len(male_images_names), len(female_images_names)])):
                male_image = Image.open(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/{male_images_names[i]}"
                )
                female_image = Image.open(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/{female_images_names[i]}"
                )
                with torch.no_grad():
                    inputs = processor(
                        text=text,
                        images=[male_image, female_image],
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    outputs = model(**inputs)
                    logits_per_text = outputs.logits_per_text
                    sim_probs = logits_per_text.softmax(dim=1).cpu().numpy()[0]
                    male_sim_prob, female_sim_prob = sim_probs[0], sim_probs[1]
                report_dict["model"].append(model_name)
                report_dict["activity"].append(activity)
                report_dict["text"].append(text)
                report_dict["male_image_name"].append(male_images_names[i])
                report_dict["female_image_name"].append(female_images_names[i])
                report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
                report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()


def load_alt(activities, report_dict, dataset_base_address, mode, gender):
    for model_name in CONFIG["alt"]:
        model = AltCLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        processor = AltCLIPProcessor.from_pretrained(model_name)
        for activity in activities:
            if mode == "single":
                text = f"A {gender} is {preprocess_activity(activity)}"
            else:
                text = f"A {gender} is {preprocess_activity(activity)} and a {reverse_gender(gender)} is in the scene"
            male_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/"
                )
                if name[0] != "."
            ]
            female_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/"
                )
                if name[0] != "."
            ]
            for i in range(np.min([len(male_images_names), len(female_images_names)])):
                male_image = Image.open(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/{male_images_names[i]}"
                )
                female_image = Image.open(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/{female_images_names[i]}"
                )
                with torch.no_grad():
                    inputs = processor(
                        text=text,
                        images=[male_image, female_image],
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    outputs = model(**inputs)
                    logits_per_text = outputs.logits_per_text
                    sim_probs = logits_per_text.softmax(dim=1).cpu().numpy()[0]
                    male_sim_prob, female_sim_prob = sim_probs[0], sim_probs[1]
                report_dict["model"].append(model_name)
                report_dict["activity"].append(activity)
                report_dict["text"].append(text)
                report_dict["male_image_name"].append(male_images_names[i])
                report_dict["female_image_name"].append(female_images_names[i])
                report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
                report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()


def load_flava(activities, report_dict, dataset_base_address, mode, gender):
    for model_name in CONFIG["flava"]:
        model = FlavaForPreTraining.from_pretrained(model_name).eval().to(device)
        model.eval()
        feature_extractor = FlavaFeatureExtractor.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        processor = FlavaProcessor.from_pretrained(model_name)
        for activity in activities:
            if mode == "single":
                text = f"A {gender} is {preprocess_activity(activity)}"
            else:
                text = f"A {gender} is {preprocess_activity(activity)} and a {reverse_gender(gender)} is in the scene"
            tokenized_text = tokenizer(
                text=text, return_tensors="pt", padding="max_length", max_length=77
            ).to(device)
            male_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/"
                )
                if name[0] != "."
            ]
            female_images_names = [
                name
                for name in os.listdir(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/"
                )
                if name[0] != "."
            ]
            for i in range(np.min([len(male_images_names), len(female_images_names)])):
                male_image = Image.open(
                    f"{dataset_base_address}/{activity}/{'Man' if mode=='single' else 'Man Woman'}/{male_images_names[i]}"
                )
                female_image = Image.open(
                    f"{dataset_base_address}/{activity}/{'Woman' if mode=='single' else 'Woman Man'}/{female_images_names[i]}"
                )
                with torch.no_grad():
                    text_features = (
                        model.flava.get_text_features(**tokenized_text)
                        .cpu()
                        .numpy()[:, 0, :]
                    )
                    text_features_norm = np.linalg.norm(text_features)
                    processed_male_image = feature_extractor(
                        images=male_image, return_tensors="pt"
                    ).to(device)
                    processed_female_image = feature_extractor(
                        images=female_image, return_tensors="pt"
                    ).to(device)
                    male_image_features = (
                        model.flava.get_image_features(**processed_male_image)
                        .cpu()
                        .numpy()[:, 0, :]
                    )
                    female_image_features = (
                        model.flava.get_image_features(**processed_female_image)
                        .cpu()
                        .numpy()[:, 0, :]
                    )
                    male_image_features_norm = np.linalg.norm(male_image_features)
                    female_image_features_norm = np.linalg.norm(female_image_features)
                    male_sim = (
                        (text_features @ male_image_features.T)
                        / (text_features_norm * male_image_features_norm)
                    ).item()
                    female_sim = (
                        (text_features @ female_image_features.T)
                        / (text_features_norm * female_image_features_norm)
                    ).item()
                    sim_probs = torch.tensor([male_sim, female_sim]).softmax(dim=-1)
                    male_sim_prob, female_sim_prob = (
                        sim_probs[0].item(),
                        sim_probs[1].item(),
                    )
                report_dict["model"].append(model_name)
                report_dict["activity"].append(activity)
                report_dict["text"].append(text)
                report_dict["male_image_name"].append(male_images_names[i])
                report_dict["female_image_name"].append(female_images_names[i])
                report_dict["male_sim_prob"].append(np.round(male_sim_prob, 3))
                report_dict["female_sim_prob"].append(np.round(female_sim_prob, 3))
        del model
        torch.cuda.empty_cache()
        gc.collect()
