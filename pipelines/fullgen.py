import json
import pandas as pd
from generators.text.diversity import sample_from_clusters
from generators.text.api import generate as generate_prompt
from tqdm import tqdm


PROMPT_PER_SCENARIO = 20


def fire(loader, opath):
    for subj, objc, activity, mode in loader:
        subj = subj[0]
        objc = objc[0]
        activity = activity[0]
        modes = mode[0].split("|")

        prompts_expected_double = []
        prompts_expected_single = []
        prompts_unexpected_double = []
        prompts_unexpected_single = []

        # Expected Double
        if "double_exp" in modes:
            base_prompt = f"a {subj} is {activity} and a {objc} is doing something else"
            for comb in tqdm(
                sample_from_clusters(PROMPT_PER_SCENARIO), desc="Expected Double"
            ):
                prompts_expected_double.append(
                    generate_prompt(base_prompt, prompts_expected_double, comb)
                )
        # UnExpected Double
        if "double_unx" in modes:
            base_prompt = f"a {objc} is {activity} and a {subj} is doing something else"
            for comb in tqdm(
                sample_from_clusters(PROMPT_PER_SCENARIO), desc="UnExpected Double"
            ):
                prompts_unexpected_double.append(
                    generate_prompt(base_prompt, prompts_unexpected_double, comb)
                )
        # Expected Single
        if "single_exp" in modes:
            base_prompt = f"a {subj} is {activity} [Single {subj.lower().capitalize()}]"
            for comb in tqdm(
                sample_from_clusters(PROMPT_PER_SCENARIO), desc="Expected Single"
            ):
                prompts_expected_single.append(
                    generate_prompt(base_prompt, prompts_unexpected_single, comb)
                )
        # UnExpected Single
        if "single_unx" in modes:
            base_prompt = f"a {objc} is {activity} [Single {objc.lower().capitalize()}]"
            for comb in tqdm(
                sample_from_clusters(PROMPT_PER_SCENARIO), desc="UnExpected Single"
            ):
                prompts_unexpected_single.append(
                    generate_prompt(base_prompt, prompts_unexpected_single, comb)
                )

        data = dict()
        if "single_unx" in modes:
            data["unx_sgl"] = prompts_unexpected_single
        if "single_exp" in modes:
            data["exp_sgl"] = prompts_expected_single
        if "double_unx" in modes:
            data["unx_dbl"] = prompts_unexpected_double
        if "double_exp" in modes:
            data["exp_dbl"] = prompts_expected_double

        pd.DataFrame(data).to_csv(opath + f'/{"_".join(activity.strip().split())}.csv')
