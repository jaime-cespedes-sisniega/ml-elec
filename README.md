# ML Pipeline
This repository aims to generate a simple Machine Learning pipeline using an open dataset with data collected from the electricity prices and demand in various areas of Australia between 7 May 1996 to 5 December 1998. The dataset is intended to predict whether the price will rise or fall in 30 minute periods. More information about the dataset is available at [Australian Electricity Market](https://www.openml.org/d/151).

## Reproducibility

In order to facilitate the reproducibility of the pipeline, the file train_pipeline.py has been implemented, which by default attempts to download the dataset from https://www.openml.org/d/151. The data is downloaded into `data/raw`, then split into to later split it into train and test, perform a small preprocessing step and store both files in `data/processed`.

The rest of the script is responsible for generating a model using the package `ml_pipeline` as a wrapper. This package is intended to be used by the serialized model outside the project when using it in the inference phase.

> **_NOTE_**: the serialized model is generated in `models` directory, but it should be stored in some model registry or in some storage that allows its consumption from outside the project. The only solution for the moment would be to copy the serialized file wherever you want to consume it. After that, you could use the `ml_pipeline` package in the same project.

Through the config.ini file different parameters related to the pipeline can be configured. For example, you can disable the download of raw data in case it has already been downloaded previously, change the ratio of training samples, use the test set with the generated model to measure its performance, etc.

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

As mentioned before, the `ml_pipeline` wrapper package can be used outside the project (`setup.py` installs that package) just by referencing it. It is assumed that the serialized model is also in the same project.
```bash
pip install git+https://github.com/jaime-cespedes-sisniega/ml-elec.git@v0.1.2
```

### Optional

To remove all reproducibility steps (data, generated model and virtualenv), the following command can be executed.
```bash
make clean_all
```
