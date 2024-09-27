import ast
import torch
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torchvision.transforms as transforms
from utils.load import Loader
from experiments.config.models import CONFIG, MODEL_NAME_MAPPER
from experiments.activity_retrieval.config.near_actions import NEAR_ACTIONS
from experiments.activity_retrieval.config.expected import EXPECTED
import gc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = CONFIG


def tendency_experiment(loader, prompts):
    result = []
    for api in tqdm(MODEL_NAMES):
        handler = MODEL_NAMES[api]["handler"]
        pbar = tqdm(MODEL_NAMES[api]["models"])
        for args in pbar:
            pbar.set_description(f"{api}_{' '.join(args)}")
            model = handler(DEVICE, *args)
            total_prompts = prompts
            for label, image in loader:
                inputs = transforms.ToPILImage()(image[0])
                probs = model(inputs, total_prompts)[0]
                result.append(
                    [
                        f"{api}_{' '.join(args)}",
                        prompts[int(np.argmax(probs))],
                        label,
                        int(np.argmax(probs)),
                        list(probs),
                    ]
                )
            del model
            torch.cuda.empty_cache()
            gc.collect()
    df = pd.DataFrame(result)
    df.columns = ["model", "prompt", "image", "prob", "probs"]
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


def make_prompt(prompt_list):
    man_single_list = []
    man_double_list = []
    woman_single_list = []
    woman_double_list = []
    for action in prompt_list:
        man_single_list.append(f"a man is {action}")
        man_double_list.append(f"a man is {action} and a woman is in the scene")
        woman_single_list.append(f"a woman is {action}")
        woman_double_list.append(f"a woman is {action} and a man is in the scene")
    return man_single_list, man_double_list, woman_single_list, woman_double_list


def experiment(gpath, opath):
    for phaze in NEAR_ACTIONS:
        source_path = f"{gpath}/{phaze}/Phaze-{phaze[-1]}"
        print(source_path)
        action_list = sorted(list(NEAR_ACTIONS[phaze].keys()))
        expected_actions = EXPECTED[phaze]
        for action in action_list:
            prompt_list = NEAR_ACTIONS[phaze][action]
            prompt_list = [action] + prompt_list
            completed_prompt = make_prompt(prompt_list)

            loader_man = Loader.load_with_name(f"{source_path}/{action}/Man", 1)
            loader_woman = Loader.load_with_name(f"{source_path}/{action}/Woman", 1)
            loader_man_woman = Loader.load_with_name(
                f"{source_path}/{action}/Man Woman", 1
            )
            loader_woman_man = Loader.load_with_name(
                f"{source_path}/{action}/Woman Man", 1
            )

            categories = {
                "man_binding": tendency_experiment(loader_man, completed_prompt[0]),
                "woman_binding": tendency_experiment(loader_woman, completed_prompt[2]),
                "man_woman_binding": tendency_experiment(
                    loader_man_woman, completed_prompt[1]
                ),
                "man_woman_2_binding": tendency_experiment(
                    loader_man_woman, completed_prompt[0]
                ),
                "woman_man_binding": tendency_experiment(
                    loader_woman_man, completed_prompt[3]
                ),
                "woman_man_2_binding": tendency_experiment(
                    loader_woman_man, completed_prompt[2]
                ),
            }

            column_names = ["model", "Recall@1", "Recall@3", "MRR", "action"]
            dfExpectedTwo = pd.DataFrame(columns=column_names)
            dfExpectedTwo2 = pd.DataFrame(columns=column_names)
            dfUnExpectedTwo = pd.DataFrame(columns=column_names)
            dfUnExpectedTwo2 = pd.DataFrame(columns=column_names)
            dfUnExpectedSingle = pd.DataFrame(columns=column_names)
            dfExpectedSingle = pd.DataFrame(columns=column_names)
            dfSingle = pd.DataFrame(columns=column_names)
            dfTwo = pd.DataFrame(columns=column_names)
            dfTwo2 = pd.DataFrame(columns=column_names)

            for action in action_list:
                for category in categories:
                    df = categories[category]
                    for model, group in df.groupby("model"):
                        correct_predictions = 0
                        total_predictions = len(group) // 2

                        listProbs = []
                        for val in group["probs"]:
                            listProbs.append(val)

                        recall_at_1, recall_at_3, mrr = calculate_metrics(listProbs)
                        if (
                            category == "woman_man_2_binding"
                            or category == "man_woman_2_binding"
                        ):
                            if category in expected_actions[action]:
                                dfExpectedTwo2.loc[len(dfExpectedTwo2)] = [
                                    model,
                                    recall_at_1,
                                    recall_at_3,
                                    mrr,
                                    action,
                                ]
                            else:
                                dfUnExpectedTwo2.loc[len(dfUnExpectedTwo2)] = [
                                    model,
                                    recall_at_1,
                                    recall_at_3,
                                    mrr,
                                    action,
                                ]
                            dfTwo2.loc[len(dfTwo2)] = [
                                model,
                                recall_at_1,
                                recall_at_3,
                                mrr,
                                action,
                            ]
                        if (
                            category == "woman_man_binding"
                            or category == "man_woman_binding"
                        ):
                            if category in expected_actions[action]:
                                dfExpectedTwo.loc[len(dfExpectedTwo)] = [
                                    model,
                                    recall_at_1,
                                    recall_at_3,
                                    mrr,
                                    action,
                                ]
                            else:
                                dfUnExpectedTwo.loc[len(dfUnExpectedTwo)] = [
                                    model,
                                    recall_at_1,
                                    recall_at_3,
                                    mrr,
                                    action,
                                ]
                            dfTwo.loc[len(dfTwo)] = [
                                model,
                                recall_at_1,
                                recall_at_3,
                                mrr,
                                action,
                            ]
                        if category == "man_binding" or category == "woman_binding":
                            if category in expected_actions[action]:
                                dfExpectedSingle.loc[len(dfExpectedSingle)] = [
                                    model,
                                    recall_at_1,
                                    recall_at_3,
                                    mrr,
                                    action,
                                ]
                            else:
                                dfUnExpectedSingle.loc[len(dfUnExpectedSingle)] = [
                                    model,
                                    recall_at_1,
                                    recall_at_3,
                                    mrr,
                                    action,
                                ]
                            dfSingle.loc[len(dfSingle)] = [
                                model,
                                recall_at_1,
                                recall_at_3,
                                mrr,
                                action,
                            ]

            out = f"{opath}/Activity-Retrieval-Agg-Single-Phaze-{phaze}"
            os.makedirs(out, exist_ok=True)

            dfSingle[dfSingle["model"].isin(MODEL_NAME_MAPPER.keys())].groupby(
                ["model"]
            )[["MRR", "Recall@1", "Recall@3"]].mean().reset_index().to_csv(
                f"{out}/{action}.csv",
                index=False,
            )

            out = f"{opath}/Activity-Retrieval-Agg-Double-Phaze-{phaze}"
            os.makedirs(out, exist_ok=True)

            dfTwo[dfTwo["model"].isin(MODEL_NAME_MAPPER.keys())].groupby(["model"])[
                ["MRR", "Recall@1", "Recall@3"]
            ].mean().reset_index().to_csv(
                f"{out}/{action}.csv",
                index=False,
            )
