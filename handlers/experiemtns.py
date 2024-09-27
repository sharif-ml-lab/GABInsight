from utils.load import Loader

from experiments.activity_retrieval.experiment import (
    experiment as activity_retrieval_exp,
)
from experiments.bias.experiment import experiment as bias_exp
from experiments.text_encoder_bias.experiment import experiment as text_encoder_bias_exp
from experiments.text_image_retrieval.experiment import (
    experiment as text_image_retrieval_exp,
)


def activity_retrieval_experiment(gpath, opath):
    activity_retrieval_exp(gpath, opath)


def bias_experiment(gpath, opath):
    bias_exp(gpath, opath)


def text_encoder_bias_experiment(opath):
    text_encoder_bias_exp(opath)


def text_image_retrieval_experiment(gpath, opath):
    text_image_retrieval_exp(gpath, opath)
