
# Tile-level Histopathology image Understanding benchmark

<img src="docs/banner.svg" />

<div align="center">

[![Python application](https://github.com/MICS-lab/thunder/actions/workflows/ci.yml/badge.svg)](https://github.com/MICS-lab/thunder/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mics-lab.github.io/thunder/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://github.com/MICS-Lab/thunder/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)

</div>

We introduce **THUNDER**, a comprehensive benchmark designed to rigorously compare foundation models across various downstream tasks in computational pathology. THUNDER enables the evaluation and analysis of feature representations, robustness, and uncertainty quantification of these models across different datasets. Our benchmark encompasses a diverse collection of well-established datasets, covering multiple cancer types, image magnifications, and varying image and sample sizes. We propose an extensive set of tasks aimed at thoroughly assessing the capabilities and limitations of foundation models in digital pathology.


## Overview

We propose a benchmark to compare and study foundation models across three axes: (i) downstream task performance, (ii) feature space comparisons, and (iii) uncertainty and robustness. Our current version integrates 23 foundation models, vision-only, vision-language, trained on pathology or natural images, on 16 datasets covering different magnifications and organs. THUNDER also supports the use of new user-defined models for direct comparisons.

<img src="docs/overview.svg" />


## Usage

An API and command line interface (CLI) are provided to allow users to download datasets, models, and run benchmarks. The API is designed to be user-friendly and allows for easy integration into existing workflows. The CLI provides a convenient way to access the same functionality from the command line.

> [!IMPORTANT]
> **Downloading supported foundation models**: you will have to visit the Huggingface URL of supported models you wish to use in order to accept usage conditions.

### API Usage
When using the API you can run the following code to download datasets, models and run a benchmark:

```python
from thunder import benchmark

benchmark("phikon", "break_his", "knn")
```

### CLI Usage
When using the CLI you can run the following command to see all available options,

```console
thunder --help
```

In order to reproduce the above example you can run the following command:

```console
thunder benchmark phikon break_his knn
```

## Installing thunder

Code tested with Python 3.10. To replicate, you can create the following conda environment and activate it,
```console
conda create -n thunder_env python=3.10
conda activate thunder_env
```

To install `thunder` run the following command:

```console
pip install -e . # install the package in editable mode
pip install . # install the package
```

Before running `thunder`, ensure that the environment variable `THUNDER_BASE_DATA_FOLDER` is defined. This variable specifies the path where outputs, foundation models, and datasets will be stored. You can set it by running:

```console
export THUNDER_BASE_DATA_FOLDER="/path/to/your/data/folder"
```

Replace `/path/to/your/data/folder` with your desired storage directory.
