from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.proto import trainer_pb2, evaluator_pb2, pusher_pb2
from tfx.utils.dsl_utils import csv_input


def get_examples(data_root):
    examples = csv_input(data_root)
    return CsvExampleGen(input_base=examples)


def get_statistics_gen(examples):
    return StatisticsGen(input_data=examples.outputs.examples)


def get_schema_gen(statistics_gen):
    return SchemaGen(
        stats=statistics_gen.outputs.output, infer_feature_shape=False)


def get_example_validator(statistics_gen, schema_gen):
    return ExampleValidator(
        stats=statistics_gen.outputs['output'],
        schema=schema_gen.outputs['output'])


def get_transform(examples, schema_gen, module_file):
    return Transform(
        input_data=examples.outputs.examples,
        schema=schema_gen.outputs.output,
        module_file=module_file)


def get_trainer(transform, schema_gen, module_file, train_steps=10000, eval_steps=5000):
    return Trainer(
        module_file=module_file,
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['output'],
        transform_output=transform.outputs['transform_output'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))


def get_model_evaluator(examples, trainer):
    return Evaluator(
        examples=examples.outputs['examples'],
        model_exports=trainer.outputs['output'],
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]))


def get_model_validator(examples, trainer):
    return ModelValidator(
        examples=examples.outputs['examples'], model=trainer.outputs['output'])


def get_model_pusher(trainer, model_validator, serving_model_dir):
    return Pusher(
        model_export=trainer.outputs['output'],
        model_blessing=model_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
