import random
import pickle
import torch
import logging
import itertools
import contextlib
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.models.sentence import MiniLMEncoder


skin_colors_men = [
    "olive-skinned woman",
    "fair-skinned man",
    "brown-skinned man",
    "dark-skinned man",
]
skin_colors_women = [
    "fair-skinned man",
    "olive-skinned woman",
    "brown-skinned woman",
    "dark-skinned woman",
]
dress_types_women = [
    "Casual wear for women",
    "Formal attire for women",
    "Traditional attire for women",
    "Activewear for women",
    "Business casual for women",
]
dress_types_men = [
    "Casual wear for men",
    "Formal attire for men",
    "Traditional attire for men",
    "Activewear for men",
    "Business casual for men",
]
financial_situations = [
    "Economically disadvantaged",
    "Living comfortably",
    "Financially aspiring",
    "Financial independence",
]
ages_man = ["Teenager", "Young adult", "Adult", "Middle-aged", "Senior"]
ages_woman = ["Teenager", "Young adult", "Adult", "Middle-aged", "Senior"]


def sample_from_clusters(total_samples):
    try:
        clustered_data, kmeans = load_cluster_data(
            filename="utils/data/text/cluster.pkl"
        )
    except:
        logging.error(
            "RUN --space genai --method config --task llm-diversity --data text"
        )
        return
    num_clusters = len(clustered_data)
    samples_per_cluster = max(total_samples // num_clusters, 1)
    sampled_combinations = []
    for cluster in clustered_data:
        if cluster:
            sampled_combinations.extend(
                random.sample(cluster, min(samples_per_cluster, len(cluster)))
            )
    remaining_samples = total_samples - len(sampled_combinations)
    while remaining_samples > 0:
        for cluster in clustered_data:
            if remaining_samples == 0:
                break
            if cluster:
                sampled_combinations.append(random.choice(cluster))
                remaining_samples -= 1
    return sampled_combinations


def find_optimal_clusters(X, max_clusters):
    best_silhouette = -1
    best_n_clusters = 2
    best_kmeans = None
    for n_clusters in range(5, min(max_clusters, X.shape[0])):
        kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X, labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters
            best_kmeans = kmeans
    return best_n_clusters, best_kmeans


def save_cluster_data(filename="utils/data/text/cluster.pkl"):
    combinations = list(
        itertools.product(
            skin_colors_men,
            dress_types_men,
            ages_man,
            skin_colors_women,
            dress_types_women,
            ages_woman,
            financial_situations,
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = MiniLMEncoder(device)
    encoded_combinations = []
    for i in tqdm(range(len(combinations)), desc="Calculating Embedding"):
        with contextlib.redirect_stdout(None):
            encoded_combinations.append(
                encoder(",".join(combinations[i])).cpu().numpy()
            )

    encoded_combinations = np.array(encoded_combinations)
    num_clusters, kmeans = find_optimal_clusters(encoded_combinations, 30)

    clustered_combinations = [[] for _ in range(num_clusters)]
    for comb, label in zip(combinations, kmeans.labels_):
        clustered_combinations[label].append(comb)

    with open(filename, "wb") as f:
        pickle.dump((clustered_combinations, kmeans), f)


def load_cluster_data(filename="utils/data/text/cluster.pkl"):
    with open(filename, "rb") as f:
        clustered_data, kmeans_model = pickle.load(f)
    return clustered_data, kmeans_model
