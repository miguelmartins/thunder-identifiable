## Installation

In order to use the package, you need to install it first. You can do this by running the following command in your terminal:

```console
pip install git+https://github.com/MICS-Lab/thunder.git
```

or you can clone the repository and install it locally like so:

```console
pip install .
```

The package is storing all the datasets, models and results under a folder that you will need to define through the environment variable `THUNDER_BASE_DATA_FOLDER`. You can do this by running the following command in your terminal:
```console
export THUNDER_BASE_DATA_FOLDER=/path/to/thunder_base_data_folder
```

!!!important
    Without this environment variable, the package will not work. The folder should be empty and the package will create the necessary subfolders.

## CLI Usage

You can run the following command to see all available options,
```console
> thunder --help
 Usage: thunder [OPTIONS] COMMAND [ARGS]...
```

The available commands are:  
- `benchmark`: Benchmarks the models on the datasets for a task.  
- `download-datasets`: Downloads datasets.  
- `download-models`: Downloads models.  
- `generate-data-splits`: Generate data splits for the downloaded datasets.  
- `results-summary`: Compiles a summary csv file of the results.


To benchmark the models, you can run the following command,
```console
> thunder benchmark --help
Usage: thunder benchmark [OPTIONS] MODEL DATASET TASK
> thunder benchmark phikon ccrcc knn
```

In case you want to download a datasets, you can run the following command,
```console
> thunder download-datasets ccrcc patch_camelyon bach
> thunder download-datasets classification
> thunder download-datasets all --make_splits # Generates splits after downloading
```

To download the models, you can run the following command,
```console
> thunder download-models phikon keep
> thunder download-models dinov2base
```

To generate splits for the downloaded, you can run the following command,
```console
> thunder generate-data-splits ccrcc patch_camelyon bach
> thunder generate-data-splits classification
> thunder generate-data-splits all
```

## API Usage

You can also use the package as a library. For example, you can run the following code to download datasets,
```python
from thunder import download_datasets, download_models, generate_splits, benchmark

# Download datasets
download_datasets(["ccrcc", "patch_camelyon", "bach"])
download_datasets(["all"])
download_datasets(["classification"])

# Download models
download_models(["phikon", "dinov2base"])

# Generate data splits
generate_splits(["all"])

# Benchmark
benchmark(model="phikon", dataset="ccrcc", task="knn")
```
