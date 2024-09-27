import torch
import numpy as np
from scipy import linalg
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_learned_perceptual_diversity(loader):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(DEVICE)

    size = len(loader)
    similarity = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating LPIPS"):
        image1 = loader.dataset[i].unsqueeze(0)
        image2 = loader.dataset[i - 1].unsqueeze(0)
        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)
        similarity[i] = float(lpips(image1, image2))

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
