import os

from pipelines.config import Config
from . import components


def get_config(pipeline_name):
    module_location = __file__[0:__file__.rfind("/")]
    Config.PIPELINE_NAME = pipeline_name

    # Config.PROJECT_ROOT =
    Config.DATA_ROOT = os.path.join(module_location, "data")

    Config.MODULE_FILE = os.path.join(module_location, "utils.py")

    Config.SERVING_MODEL_DIR = os.path.join(os.getcwd(), "servings", pipeline_name)
    return Config
