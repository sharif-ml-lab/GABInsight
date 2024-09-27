import ast
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torchvision.transforms as transforms
from utils.load import Loader
from experiments.config.models import CONFIG, MODEL_NAME_MAPPER
from experiments.config.actions import ACTIONS
from experiments.bias.config.assigns import TRUE_ASSIGNS, FALSE_ASSIGNS
from experiments.bias.config.expected import EXPECTED
import gc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = CONFIG


def tendency_experiment(loader, prompt, neg_prompt):
    result = []
    for api in tqdm(MODEL_NAMES):
        handler = MODEL_NAMES[api]["handler"]
        pbar = tqdm(MODEL_NAMES[api]["models"])
        for args in pbar:
            pbar.set_description(f"{api}_{' '.join(args)}")
            model = handler(DEVICE, *args)
            total_prompts = [prompt, neg_prompt]
            for label, image in loader:
                inputs = transforms.ToPILImage()(image[0])
                prob_pos, prob_neg = model(inputs, total_prompts)[0]
                result.append([f"{api}_{' '.join(args)}", prompt, label, prob_pos])
                result.append([f"{api}_{' '.join(args)}", neg_prompt, label, prob_neg])
            del model
            torch.cuda.empty_cache()
            gc.collect()
    df = pd.DataFrame(result)
    df.columns = ["model", "prompt", "image", "prob"]
    return df


def calculate_metrics(prob_lists):
    recall_at_1 = 0
    recall_at_3 = 0
    mrr = 0.0

    for probs in prob_lists:
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        correct_rank = sorted_indices.index(0) + 1
        if correct_rank == 1:
            recall_at_1 += 1
        if correct_rank <= 3:
            recall_at_3 += 1

        mrr += 1 / correct_rank

    num_lists = len(prob_lists)
    recall_at_1 /= num_lists
    recall_at_3 /= num_lists
    mrr /= num_lists

    return recall_at_1, recall_at_3, mrr


def experiment(gpath, opath):
    for phaze, expected in ACTIONS.items():
        action_list = sorted(ACTIONS[phaze]["man"] + ACTIONS[phaze]["woman"])
        source_path = f"{gpath}/{phaze}/Phaze-{phaze[-1]}"
        for action in action_list:
            loader_man = Loader.load_with_name(f"{source_path}/{action}/Man", 1)
            loader_woman = Loader.load_with_name(f"{source_path}/{action}/Woman", 1)
            loader_man_woman = Loader.load_with_name(
                f"{source_path}/{action}/Man Woman", 1
            )
            loader_woman_man = Loader.load_with_name(
                f"{source_path}/{action}/Woman Man", 1
            )

            categories = {
                "man": tendency_experiment(
                    loader_man,
                    TRUE_ASSIGNS[phaze][action]["man"],
                    FALSE_ASSIGNS[phaze][action]["man"],
                ),
                "woman": tendency_experiment(
                    loader_woman,
                    TRUE_ASSIGNS[phaze][action]["woman"],
                    FALSE_ASSIGNS[phaze][action]["woman"],
                ),
                "man_woman": tendency_experiment(
                    loader_man_woman,
                    TRUE_ASSIGNS[phaze][action]["man_woman"],
                    FALSE_ASSIGNS[phaze][action]["man_woman"],
                ),
                "man_woman_2": tendency_experiment(
                    loader_man_woman,
                    TRUE_ASSIGNS[phaze][action]["man_woman_2"],
                    FALSE_ASSIGNS[phaze][action]["man_woman_2"],
                ),
                "woman_man": tendency_experiment(
                    loader_woman_man,
                    TRUE_ASSIGNS[phaze][action]["woman_man"],
                    FALSE_ASSIGNS[phaze][action]["woman_man"],
                ),
                "woman_man_2": tendency_experiment(
                    loader_woman_man,
                    TRUE_ASSIGNS[phaze][action]["woman_man_2"],
                    FALSE_ASSIGNS[phaze][action]["woman_man_2"],
                ),
            }

            column_names = ["model", "acc", "action", "type"]
            dfExpectedTwo = pd.DataFrame(columns=column_names)
            dfExpectedTwo2 = pd.DataFrame(columns=column_names)
            dfUnExpectedTwo = pd.DataFrame(columns=column_names)
            dfUnExpectedTwo2 = pd.DataFrame(columns=column_names)
            dfUnExpectedSingle = pd.DataFrame(columns=column_names)
            dfExpectedSingle = pd.DataFrame(columns=column_names)

            for key in TRUE_ASSIGNS[phaze].keys():
                for class_type in TRUE_ASSIGNS[phaze][key].keys():
                    df = categories[class_type]
                    for model, group in df.groupby("model"):
                        correct_predictions = 0
                        total_predictions = len(group) // 2
                        for _, image_group in group.groupby("image"):
                            try:
                                correct_prediction = (
                                    image_group[
                                        image_group["prompt"]
                                        == TRUE_ASSIGNS[phaze][key][class_type]
                                    ]["prob"].values[0]
                                    > image_group[
                                        image_group["prompt"]
                                        == FALSE_ASSIGNS[phaze][key][class_type]
                                    ]["prob"].values[0]
                                )
                            except Exception as e:
                                pass
                            if correct_prediction:
                                correct_predictions += 1
                        accuracy = correct_predictions / total_predictions
                        if class_type == "woman_man_2" or class_type == "man_woman_2":
                            if class_type in EXPECTED[phaze][key]:
                                dfExpectedTwo2.loc[len(dfExpectedTwo2)] = [
                                    model,
                                    accuracy,
                                    key,
                                    class_type,
                                ]
                            else:
                                dfUnExpectedTwo2.loc[len(dfUnExpectedTwo2)] = [
                                    model,
                                    accuracy,
                                    key,
                                    class_type,
                                ]
                        if class_type == "woman_man" or class_type == "man_woman":
                            if class_type in EXPECTED[phaze][key]:
                                dfExpectedTwo.loc[len(dfExpectedTwo)] = [
                                    model,
                                    accuracy,
                                    key,
                                    class_type,
                                ]
                            else:
                                dfUnExpectedTwo.loc[len(dfUnExpectedTwo)] = [
                                    model,
                                    accuracy,
                                    key,
                                    class_type,
                                ]
                        if class_type == "man" or class_type == "woman":
                            if class_type in EXPECTED[phaze][key]:
                                dfExpectedSingle.loc[len(dfExpectedSingle)] = [
                                    model,
                                    accuracy,
                                    key,
                                    class_type,
                                ]
                            else:
                                dfUnExpectedSingle.loc[len(dfUnExpectedSingle)] = [
                                    model,
                                    accuracy,
                                    key,
                                    class_type,
                                ]

            (
                dfExpectedTwo[dfExpectedTwo["model"].isin(MODEL_NAME_MAPPER.keys())]
                .groupby(["model"])["acc"]
                .mean()
                .reset_index()
            ).to_csv(f"{opath}/Bias-Expected-Single-Agg-{phaze}-{action}.csv")

            (
                dfUnExpectedTwo[dfUnExpectedTwo["model"].isin(MODEL_NAME_MAPPER.keys())]
                .groupby(["model"])["acc"]
                .mean()
                .reset_index()
            ).to_csv(f"{opath}/Bias-Unexpected-Double-Agg-{phaze}-{action}.csv")

            (
                dfUnExpectedSingle[
                    dfUnExpectedSingle["model"].isin(MODEL_NAME_MAPPER.keys())
                ]
                .groupby(["model"])["acc"]
                .mean()
                .reset_index()
            ).to_csv(f"{opath}/Bias-Unexpected-Single-Agg-{phaze}-{action}.csv")
