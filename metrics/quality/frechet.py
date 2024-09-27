import torch
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

fid = FrechetInceptionDistance(feature=64)


def calculate_frechet_inception_distance(loader_real, loader_fake):
    real_features = get_images(loader_real)
    fake_features = get_images(loader_fake)

    fid.update(real_features, real=True)
    fid.update(fake_features, real=False)

    return float(fid.compute().cpu().numpy())


def get_images(loader):
    features = []
    for data_batch in tqdm(loader, desc="Image Expansion"):
        features.append(data_batch[0].squeeze(0).cpu())
    features = torch.stack(features, dim=0).to(torch.uint8).cpu()
    return features
