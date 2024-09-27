import os
import pandas as pd
from experiments.text_encoder_bias.models import *
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


def experiment(opath):
    for phaze in ACTIONS:
        activities = sorted(ACTIONS[phaze]["man"] + ACTIONS[phaze]["woman"])
        report_dict = {
            "model": [],
            "activity": [],
            "male_sim_prob": [],
            "female_sim_prob": [],
        }

        try:
            load_open_clip(activities, report_dict)
        except Exception as e:
            print(str(e))
            pass

        try:
            load_align(activities, report_dict)
        except Exception as e:
            print(str(e))
            pass

        try:
            load_alt(activities, report_dict)
        except Exception as e:
            print(str(e))
            pass

        try:
            load_flava(activities, report_dict)
        except Exception as e:
            print(str(e))
            pass

        text_bias_report = pd.DataFrame(data=report_dict)
        text_bias_report["male_column"] = text_bias_report["male_sim_prob"].apply(
            lambda x: 1 if x > 0.5 else 0
        )
        text_bias_report["female_column"] = text_bias_report["female_sim_prob"].apply(
            lambda x: 1 if x > 0.5 else 0
        )

        male_report = text_bias_report[
            ~text_bias_report["activity"].isin(ACTIONS[phaze]["woman"])
        ]

        male_report_deleted = male_report[
            male_report["model"].isin(MODEL_NAME_MAPPER.keys())
        ]
        female_report = text_bias_report[
            ~text_bias_report["activity"].isin(ACTIONS[phaze]["man"])
        ]
        female_report_deleted = female_report[
            female_report["model"].isin(MODEL_NAME_MAPPER.keys())
        ]

        female_report_deleted_renamed = female_report_deleted.rename(
            columns={"female_column": "Correct"}, errors="raise"
        )
        male_report_deleted_renamed = male_report_deleted.rename(
            columns={"male_column": "Correct"}, errors="raise"
        )

        pd.concat(
            [female_report_deleted_renamed, male_report_deleted_renamed],
            ignore_index=True,
        ).groupby(["model"])["Correct"].mean().reset_index().to_csv(
            f"{opath}/Text-Encoder-Bias-{phaze}.csv"
        )
