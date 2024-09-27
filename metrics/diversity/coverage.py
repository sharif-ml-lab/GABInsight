import torch
import numpy as np
from prdc import compute_prdc
import torchvision.transforms as transforms
from utils.models.embedding import ViTLarge
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REAL_FEATURES_FILE = "real_features.npy"


def store_real_features(loader_real, embedding_model):
    real_features = []
    for data_batch in tqdm(loader_real, desc="Store Embedding Real"):
        with torch.no_grad():
            inputs = transforms.ToPILImage()(data_batch[0])
            features = embedding_model(inputs)
        real_features.append(features.cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    np.save(REAL_FEATURES_FILE, real_features)
    return real_features


def load_real_features(loader_real, embedding_model):
    if os.path.exists(REAL_FEATURES_FILE):
        print("Loading real features from file.")
        real_features = np.load(REAL_FEATURES_FILE)
    else:
        print("Computing and storing real features.")
        real_features = store_real_features(loader_real, embedding_model)
    return real_features


def sample_real_features(real_features, num_samples):
    if len(real_features) > num_samples:
        indices = np.random.choice(len(real_features), num_samples, replace=False)
        sampled_real_features = real_features[indices]
    else:
        sampled_real_features = real_features
    return sampled_real_features


def calculate_image_diversity(loader_real, loader_fake):
    embedding_model = ViTLarge(DEVICE)

    real_features = load_real_features(loader_real, embedding_model)

    fake_features = []
    for data_batch in tqdm(loader_fake, desc="Store Embedding Fake"):
        with torch.no_grad():
            inputs = transforms.ToPILImage()(data_batch[0])
            features = embedding_model(inputs)
        fake_features.append(features.cpu().numpy())

    fake_features = np.concatenate(fake_features, axis=0)
    real_features = sample_real_features(real_features, len(fake_features))
    nearest_k = 5
    metrics = compute_prdc(
        real_features=real_features, fake_features=fake_features, nearest_k=nearest_k
    )
    return metrics
