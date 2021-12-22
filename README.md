# ML Pipeline
This repository aims to generate a simple Machine Learning pipeline using an open dataset with data collected from the electricity prices and demand in various areas of Australia between 7 May 1996 to 5 December 1998. The dataset is intended to predict whether the price will rise or fall in 30 minute periods. More information about the dataset is available at [Australian Electricity Market](https://www.openml.org/d/151).

## Reproducibility

In order to facilitate the reproducibility of the pipeline, the file train_pipeline.py has been implemented, which by default attempts to download the dataset from https://www.openml.org/d/151. The data is downloaded into `data/raw`, then split into to later split it into train and test, perform a small preprocessing step and store both files in `data/processed`.

The rest of the script is responsible for generating a model using the package `ml_pipeline` as a wrapper. This package is intended to be used by the serialized model outside the project when using it in the inference phase.

The model will be stored in a MongoDB that acts as a model registry.

Through the `config.ini` file different parameters related to the pipeline can be configured. For example, you can disable the download of raw data in case it has already been downloaded previously, change the ratio of training samples, use the test set with the generated model to measure its performance, etc. Model registry configuration can also be set in this file.

## Usage

The following command allows to create a virtualenv and install the requirements.

```bash
make install
```

### Pipeline

In order to generate the pipeline, from getting to data from the original source, preprocess it and generate a model, the following command can be run.
```bash
make pipeline
```

### Inference

As mentioned before, the `ml_pipeline` wrapper package can be used outside the project (`setup.py` installs that package) just by referencing it.
```bash
pip install git+https://github.com/jaime-cespedes-sisniega/ml-elec.git@v0.1.2
```
