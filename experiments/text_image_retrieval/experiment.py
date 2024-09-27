import os
import pandas as pd
from experiments.text_image_retrieval.models import *
from experiments.config.actions import ACTIONS


MODEL_NAME_MAPPER = {
    "kakaobrain/align-base": "Align",
    "BAAI/AltCLIP": "AltCLIP",
    "facebook/flava-full": "Flava",
    "EVA01-g-14 laion400m_s11b_b41k": "EVA01-g-14",
    "EVA02-L-14 merged2b_s4b_b131k": "EVA02-L-14",
    "RN50x64 openai": "CLIP-RN50x64",
    "ViT-B-16 openai": "CLIP-ViT-B-16",
    "ViT-B-32 negCLIP.pt": "NegCLIP-ViT-B-32",
    "ViT-B-32 openai": "CLIP-ViT-B-32",
    "ViT-L-14 openai": "CLIP-ViT-L-14",
    "coca_ViT-B-32 laion2b_s13b_b90k": "coca_ViT-B-32",
    "coca_ViT-L-14 laion2b_s13b_b90k": "coca_ViT-L-14",
}

mode_list = ["single", "double"]
gender_list = ["woman", "man", "person"]


def experiment(gpath, opath):
    for phaze in ACTIONS:
        for mode in mode_list:
            for gender in gender_list:
                dataset_base_address = f"{gpath}/{phaze}/Phaze-{phaze[-1]}"
                activities = sorted(ACTIONS[phaze]["man"] + ACTIONS[phaze]["woman"])

                report_dict = {
                    "model": [],
                    "activity": [],
                    "text": [],
                    "male_image_name": [],
                    "female_image_name": [],
                    "male_sim_prob": [],
                    "female_sim_prob": [],
                }

                try:
                    load_open_clip(
                        activities, report_dict, dataset_base_address, mode, gender
                    )
                except:
                    pass

                try:
                    load_align(
                        activities, report_dict, dataset_base_address, mode, gender
                    )
                except:
                    pass

                try:
                    load_alt(
                        activities, report_dict, dataset_base_address, mode, gender
                    )
                except:
                    pass

                try:
                    load_flava(
                        activities, report_dict, dataset_base_address, mode, gender
                    )
                except:
                    pass

                report_df = pd.DataFrame(data=report_dict)

                def calculate_score(row):
                    if gender == "person":
                        if row["activity"] in ACTIONS[phaze]["man"]:
                            return 1 if row["male_sim_prob"] >= 0.5 else 0
                        else:
                            return 0 if row["male_sim_prob"] >= 0.5 else 1
                    elif gender == "man":
                        return 1 if row["male_sim_prob"] >= 0.5 else 0
                    else:
                        return 0 if row["male_sim_prob"] >= 0.5 else 1

                report_df["score"] = report_df.apply(calculate_score, axis=1)

                text_bias_report_man = (
                    report_df.groupby(["model", "activity"])["score"]
                    .mean()
                    .reset_index()
                )

                male_report_man = text_bias_report_man[
                    ~text_bias_report_man["activity"].isin(ACTIONS[phaze]["woman"])
                ]
                male_report_deleted_man = male_report_man[
                    male_report_man["model"].isin(MODEL_NAME_MAPPER.keys())
                ]

                female_report_man = text_bias_report_man[
                    ~text_bias_report_man["activity"].isin(ACTIONS[phaze]["man"])
                ]
                female_report_deleted_man = female_report_man[
                    female_report_man["model"].isin(MODEL_NAME_MAPPER.keys())
                ]

                text_bias_report_woman = (
                    report_df.groupby(["model", "activity"])["score"]
                    .mean()
                    .reset_index()
                )

                male_report_woman = text_bias_report_woman[
                    ~text_bias_report_woman["activity"].isin(ACTIONS[phaze]["woman"])
                ]
                male_report_deleted_woman = male_report_woman[
                    male_report_woman["model"].isin(MODEL_NAME_MAPPER.keys())
                ]

                female_report_woman = text_bias_report_woman[
                    ~text_bias_report_woman["activity"].isin(ACTIONS[phaze]["man"])
                ]
                female_report_deleted_woman = female_report_woman[
                    female_report_woman["model"].isin(MODEL_NAME_MAPPER.keys())
                ]

                resultExpected = pd.concat(
                    [male_report_deleted_man, female_report_deleted_woman],
                    ignore_index=True,
                )
                resultUnExpected = pd.concat(
                    [male_report_deleted_woman, female_report_deleted_man],
                    ignore_index=True,
                )

                resultExpected.groupby(["model"])["score"].mean().reset_index().to_csv(
                    f"{opath}/Text-Image-Retrieval-{phaze}.csv", index=False
                )
                resultUnExpected.groupby(["model"])[
                    "score"
                ].mean().reset_index().to_csv(
                    f"{opath}/Text-Image-Retrieval-{phaze}.csv", index=False
                )
