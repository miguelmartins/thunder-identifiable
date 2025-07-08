import os
from typing import List

import typer
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def download_datasets(datasets: List[str], make_splits: bool = False):
    """Download datasets for the benchmark."""
    from . import download_datasets

    if "THUNDER_BASE_DATA_FOLDER" not in os.environ:
        raise EnvironmentError(
            "Please set base data directory of thunder using `export THUNDER_BASE_DATA_FOLDER=/base/data/directory`"
        )

    download_datasets(datasets, make_splits)


@app.command()
def generate_data_splits(datasets: List[str]):
    """Generate data splits for the benchmark."""
    from . import generate_splits

    if "THUNDER_BASE_DATA_FOLDER" not in os.environ:
        raise EnvironmentError(
            "Please set base data directory of thunder using `export THUNDER_BASE_DATA_FOLDER=/base/data/directory`"
        )

    generate_splits(datasets)


@app.command()
def download_models(models: List[str]):
    """Download models for the benchmark."""
    from . import download_models

    if "THUNDER_BASE_DATA_FOLDER" not in os.environ:
        raise EnvironmentError(
            "Please set base data directory of thunder using `export THUNDER_BASE_DATA_FOLDER=/base/data/directory`"
        )

    for model in models:
        download_models(model)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def benchmark(
    model: str,
    dataset: str,
    task: str,
    loading_mode: Annotated[
        str,
        typer.Option(
            help="Loading modes one of: online_loading, image_pre_loading, embedding_pre_loading"
        ),
    ] = "online_loading",
    lora: Annotated[
        bool, typer.Option(help="If provided lora adaptation will be performed")
    ] = False,
    ckpt_save_all: Annotated[
        bool,
        typer.Option(
            help="If provided all checkpoints will be saved, otherwise only the best will be kept"
        ),
    ] = False,
    online_wandb: Annotated[
        bool, typer.Option(help="Logging with the online mode of wandb")
    ] = False,
    recomp_embs: Annotated[
        bool,
        typer.Option(
            help="If provided embeddings will be re-computed even if already saved"
        ),
    ] = False,
    retrain_model: Annotated[
        bool,
        typer.Option(
            help="If provided model will be re-trained even if already trained and saved ckpts"
        ),
    ] = False,
    kwargs: Annotated[List[str], typer.Argument(help="Additional arguments")] = None,
):
    from . import benchmark

    if "THUNDER_BASE_DATA_FOLDER" not in os.environ:
        raise EnvironmentError(
            "Please set base data directory of thunder using `export THUNDER_BASE_DATA_FOLDER=/base/data/directory`"
        )

    kwargs = {} if kwargs is None else kwargs
    assert len(kwargs) % 2 == 0, "Kwargs should be in key=value format."
    kwargs = {
        kwargs[i].replace("--", ""): kwargs[i + 1] for i in range(0, len(kwargs), 2)
    }

    benchmark(
        model,
        dataset,
        task,
        loading_mode,
        lora,
        ckpt_save_all,
        online_wandb,
        recomp_embs,
        retrain_model,
        **kwargs,
    )


@app.command()
def results_summary():
    """Generate a summary of the results."""
    from . import gather_results

    if "THUNDER_BASE_DATA_FOLDER" not in os.environ:
        raise EnvironmentError(
            "Please set base data directory of thunder using `export THUNDER_BASE_DATA_FOLDER=/base/data/directory`"
        )

    gather_results()
