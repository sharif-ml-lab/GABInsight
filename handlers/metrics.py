from utils.load import Loader

from metrics.quality.inception import calculate_inception_score
from metrics.quality.frechet import calculate_frechet_inception_distance

from metrics.diversity.coverage import calculate_image_diversity
from metrics.diversity.perceptual import calculate_learned_perceptual_diversity
from metrics.diversity.ssim import calculate_structural_diversity


def inception_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_is, std_is = calculate_inception_score(generated_dataset)
    return f"Mean IS: {mean_is:.3}, SD: {std_is:.3}"


def frechet_handler(gpath, rpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    real_dataset = Loader.load(rpath, batch_size=1)
    fid_score = calculate_frechet_inception_distance(real_dataset, generated_dataset)
    return f"FID: {fid_score:.3}"


def perceptual_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1, tan_scale=True)
    mean_lpips, std_lpips = calculate_learned_perceptual_diversity(generated_dataset)
    return f"Mean LPIPS: {mean_lpips:.3}, SD: {std_lpips:.3}"


def coverage_image_handler(gpath, rpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    real_dataset = Loader.load(rpath, batch_size=1)
    metrics = calculate_image_diversity(real_dataset, generated_dataset)
    return f"Density & Converage: {metrics}"


def ssim_image_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_ssim, std_ssim = calculate_structural_diversity(generated_dataset)
    return f"Mean SSIM: {mean_ssim:.3}, SD: {std_ssim:.3}"
