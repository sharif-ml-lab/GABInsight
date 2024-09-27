from utils.models.clip import AltClip, Flava, AlignClip, ViTOpenAIClip


CONFIG = {
    "alt": {
        "models": [["BAAI/AltCLIP"]],
        "handler": AltClip,
    },
    # "flava": {
    #     "models": [["facebook/flava-full"]],
    #     "handler": Flava,
    # },
    # "align": {"models": [["kakaobrain/align-base"]], "handler": AlignClip},
    # "openai": {
    #     "models": [
    #         # ["ViT-B-32", "negCLIP.pt"], # first download and add negClip weights in this directory
    #         ["EVA01-g-14", "laion400m_s11b_b41k"],
    #         ["EVA02-L-14", "merged2b_s4b_b131k"],
    #         ["RN50x64", "openai"],
    #         ["ViT-B-16", "openai"],
    #         ["ViT-B-32", "openai"],
    #         ["ViT-L-14", "openai"],
    #         ["coca_ViT-B-32", "laion2b_s13b_b90k"],
    #         ["coca_ViT-L-14", "laion2b_s13b_b90k"],
    #     ],
    #     "handler": ViTOpenAIClip,
    # },
}


MODEL_NAME_MAPPER = {
    # "align_kakaobrain/align-base": "Align",
    "alt_BAAI/AltCLIP": "AltCLIP",
    # "flava_facebook/flava-full": "Flava",
    # "openai_EVA01-g-14 laion400m_s11b_b41k": "EVA01-g-14",
    # "openai_EVA02-L-14 merged2b_s4b_b131k": "EVA02-L-14",
    # "openai_RN50x64 openai": "CLIP-RN50x64",
    # "openai_ViT-B-16 openai": "CLIP-ViT-B-16",
    # # 'openai_ViT-B-32 negCLIP.pt':'NegCLIP-ViT-B-32', # first download and add negClip weights in this directory
    # "openai_ViT-B-32 openai": "CLIP-ViT-B-32",
    # "openai_ViT-L-14 openai": "CLIP-ViT-L-14",
    # "openai_coca_ViT-B-32 laion2b_s13b_b90k": "coca_ViT-B-32",
    # "openai_coca_ViT-L-14 laion2b_s13b_b90k": "coca_ViT-L-14",
}
