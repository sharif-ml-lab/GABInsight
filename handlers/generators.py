from utils.load import Loader

from generators.text.trigger import generate_text
from generators.text.diversity import save_cluster_data


def prompts_llm_handler(opath, model, prompt, count):
    generate_text(opath, model, prompt, count)


def llm_diversity_handler():
    save_cluster_data("utils/data/text/cluster.pkl")
