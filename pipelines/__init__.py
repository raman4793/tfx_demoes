from importlib import import_module

from tfx.orchestration import pipeline, metadata

from pipelines import config


def get_pipeline(pipeline_name):
    module = import_module("." + pipeline_name, package="pipelines")
    configuration = module.get_config(pipeline_name)
    return create_pipeline(configuration, module)


def create_pipeline(configuration, module) -> pipeline.Pipeline:
    examples = module.components.get_examples(configuration.DATA_ROOT)

    statistics_gen = module.components.get_statistics_gen(examples)

    schema_gen = module.components.get_schema_gen(statistics_gen)

    validate_stats = module.components.get_example_validator(statistics_gen, schema_gen)

    transform = module.components.get_transform(examples, schema_gen, configuration.MODULE_FILE)

    trainer = module.components.get_trainer(transform, schema_gen, configuration.MODULE_FILE)

    model_analyzer = module.components.get_model_evaluator(examples, trainer)

    model_validator = module.components.get_model_validator(examples, trainer)

    pusher = module.components.get_model_pusher(trainer, model_validator, configuration.SERVING_MODEL_DIR)

    return pipeline.Pipeline(
        pipeline_name=configuration.PIPELINE_NAME,
        pipeline_root=configuration.PIPELINE_ROOT,
        components=[
            examples, statistics_gen, schema_gen, validate_stats, transform,
            trainer, model_analyzer, model_validator, pusher
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            configuration.METADATA_PATH))
