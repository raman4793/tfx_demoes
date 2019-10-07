import os


class Config:
    PIPELINE_NAME = ""

    PROJECT_ROOT = ""
    DATA_ROOT = ""

    MODULE_FILE = ""

    SERVING_MODEL_DIR = ""

    TFX_ROOT = os.path.join(os.environ['HOME'], 'tfx')
    PIPELINE_ROOT = os.path.join(TFX_ROOT, 'pipelines', PIPELINE_NAME)

    METADATA_PATH = os.path.join(TFX_ROOT, 'metadata', PIPELINE_NAME,
                                 'metadata.db')
