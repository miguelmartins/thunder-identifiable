import os


def is_model_available(model_name: str) -> bool:
    """Check if the model is already available in the local directory.

    Args:
        model_name (str): Name of the model to check.

    Returns:
        bool: True if the model is available, False otherwise.
    """
    models_dir = os.path.join(
        os.environ["THUNDER_BASE_DATA_FOLDER"], "pretrained_ckpts/"
    )
    return os.path.exists(os.path.join(models_dir, model_name))


def load_custom_model_from_file(python_file: str):
    """
    Loading a model that inherits from the PretrainedModel abstract from a python file.

    :param python_file: path to the python file.
    """
    import importlib
    import inspect
    import sys

    from .pretrained_models import PretrainedModel

    spec = importlib.util.spec_from_file_location("custom_model", python_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Search for subclasses of MyBaseClass
    models = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, PretrainedModel) and obj is not PretrainedModel:
            models.append(obj())  # Instantiate

    if len(models) > 1:
        raise ValueError("More than one subclass of PretrainedModel found in the file.")

    model = models[0]

    if not hasattr(model, "name"):
        raise ValueError(
            "PretrainedModel Class does not have a name property that is required."
        )
    if not hasattr(model, "emb_dim"):
        raise ValueError(
            "PretrainedModel Class does not have an emb_dim property that is required."
        )

    return model
