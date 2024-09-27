from utils.load import Loader
from pipelines.fullgen import fire as full_fire


def full_generation(cpath, opath):
    pipeline_captions = Loader.load_pipe(cpath, batch_size=1)
    full_fire(pipeline_captions, opath)
