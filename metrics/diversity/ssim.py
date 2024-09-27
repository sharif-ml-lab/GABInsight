import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_structural_diversity(loader):
    size = len(loader)
    similarity = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating SSIM"):
        image1 = loader.dataset[i].unsqueeze(0)
        image2 = loader.dataset[i - 1].unsqueeze(0)
        similarity[i] = calculate_ssim(image1.cpu().numpy()[0], image2.cpu().numpy()[0])

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()


def calculate_ssim(img1, img2):
    score = ssim(
        img1,
        img2,
        multichannel=True,
        win_size=31,
        channel_axis=0,
        data_range=img1.max() - img1.min(),
    )
    return score
