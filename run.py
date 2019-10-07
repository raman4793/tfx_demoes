import argparse

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from pipelines import get_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFX pipeline demo')
    parser.add_argument('pipeline_name', type=str, nargs='+',
                        help='Supported values are iris, taxi')
    args = parser.parse_args()
    pipeline = get_pipeline(args.pipeline_name[-1])
    BeamDagRunner().run(pipeline)
