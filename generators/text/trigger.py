import os
import time
import torch
import random
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from generators.text.api import generate
from generators.text.diversity import sample_from_clusters


def generate_text(output_path, model, base_prompt, count):
    """
    Generaing Text With LLMs
    """
    prompts = []
    selected_combinations = sample_from_clusters(count)
    for combination in tqdm(selected_combinations, desc="Prompt Generation"):
        gen_prompt = generate(base_prompt, prompts, combination, model="llama2:70b")
        prompts.append(gen_prompt)
    pd.DataFrame({"caption": prompts}).to_csv(output_path, index=False)
    return prompts
