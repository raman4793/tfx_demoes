from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow.python.keras import Input
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.layers import concatenate, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow_transform.tf_metadata import schema_utils

_FEATURE_KEYS = ["userId", "movieId"]
_LABEL_KEY = "rating"


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP')


def _fill_in_missing(x):
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def preprocessing_fn(inputs):
    outputs = {}
    for key in _FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.sparse_tensor_to_dense_with_shape(inputs[key], shape=[None, 1])

    outputs[_transformed_name(_LABEL_KEY)] = tft.sparse_tensor_to_dense_with_shape(inputs[_LABEL_KEY], shape=[None, 1])

    return outputs


def _build_estimator(config):
    inputs = [Input(shape=(1,), name=f) for f in _transformed_names(_FEATURE_KEYS)]
    input_layer = concatenate(inputs)
    d1 = Dense(8, activation='relu')(input_layer)
    output = Dense(3, activation='softmax')(d1)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])
    tf.logging.info(model.summary())
    model = model_to_estimator(model, config=config)
    return model


def _example_serving_receiver_fn(tf_transform_output, schema):
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)
    transformed_features.pop(_transformed_name(_LABEL_KEY))

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output, schema):
    """Build everything needed for the tf-model-analysis to run the model.
    Args:
      tf_transform_output: A TFTransformOutput.
      schema: the schema of the input data.
    Returns:
      EvalInputReceiver function, which contains:
        - Tensorflow graph which parses raw untransformed features, applies the
          tf-transform preprocessing operators.
        - Set of raw, untransformed features.
        - Label against which predictions will be compared.
    """
    # Notice that the inputs are raw features, not transformed features here.
    raw_feature_spec = _get_raw_feature_spec(schema)

    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')

    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.parse_example(serialized_tf_example, raw_feature_spec)

    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    transformed_features = tf_transform_output.transform_raw_features(
        features)

    # The key name MUST be 'examples'.
    receiver_tensors = {'examples': serialized_tf_example}

    # NOTE: Model is driven by transformed features (since training works on the
    # materialized output of TFT, but slicing will happen on raw features.
    features = transformed_features
    labels = features.pop(_transformed_name(_LABEL_KEY))

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=labels)


def _input_fn(filenames, tf_transform_output, batch_size=200):
    """Generates features and labels for training or evaluation.
    Args:
      filenames: [str] list of CSV files to read data from.
      tf_transform_output: A TFTransformOutput.
      batch_size: int First dimension size of the Tensors returned by input_fn
    Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

    transformed_features = dataset.make_one_shot_iterator().get_next()
    # We pop the label because we do not want to use it as a feature while we're
    # training.
    return transformed_features, transformed_features.pop(
        _transformed_name(_LABEL_KEY))


# TFX will call this function
def trainer_fn(hparams, schema):
    """Build the estimator using the high level API.
    Args:
      hparams: Holds hyperparameters used to train the model as name/value pairs.
      schema: Holds the schema of the training examples.
    Returns:
      A dict of the following:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.
    """
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    train_batch_size = 40
    eval_batch_size = 40

    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

    train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
        hparams.train_files,
        tf_transform_output,
        batch_size=train_batch_size)

    eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
        hparams.eval_files,
        tf_transform_output,
        batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
        train_input_fn,
        max_steps=hparams.train_steps)

    serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
        tf_transform_output, schema)

    exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='chicago-taxi-eval')

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=999, keep_checkpoint_max=1)

    run_config = run_config.replace(model_dir=hparams.serving_model_dir)

    estimator = _build_estimator(config=run_config)

    # Create an input receiver for TFMA processing
    receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
        tf_transform_output, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }
