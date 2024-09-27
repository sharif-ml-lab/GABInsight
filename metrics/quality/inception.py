import os
import torch
import numpy as np
from torchvision import models
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_inception_score(loader, splits=10):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception_model = models.inception_v3(weights=weights).to(DEVICE)
    inception_model.eval()

    preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Calculating Inceptions Features"):
            batch = batch.to(DEVICE)
            pred = inception_model(batch)
            preds.append(pred)

    preds = torch.cat(preds, 0)

    pyx = torch.softmax(preds, 1)

    scores = []
    for i in tqdm(range(splits), desc="Calculating Inceptions Score"):
        part = pyx[i * (pyx.shape[0] // splits) : (i + 1) * (pyx.shape[0] // splits), :]
        py = part.mean(0).unsqueeze(0)
        scores.append((part * (part / py).log()).sum(1).mean().exp())

    return torch.mean(torch.tensor(scores)), torch.std(torch.tensor(scores))
