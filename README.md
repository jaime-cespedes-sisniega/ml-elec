# ML Pipeline
This repository aims to generate a simple Machine Learning pipeline using an open dataset with data collected from the electricity prices and demand in various areas of Australia between 7 May 1996 to 5 December 1998. The dataset is intended to predict whether the price will rise or fall in 30 minute periods. More information about the dataset is available at [Australian Electricity Market](https://www.openml.org/d/151).

## Settings

Modify and complete `.env.example` file to set your data, pipeline, MinIO and MLflow configuration parameters.

Rename `.env.example` to `.env`.

```bash
mv .env.example .env
```

Through the `.env` file different parameters related to the pipeline can be configured. For example, you can set hyperparameter tuning arguments, use the test set with the generated model to measure its performance, etc. Model registry configuration can also be set in this file.

The model will be stored in [MLflow´s model registry](https://github.com/mlflow/mlflow/).

## Usage

The following command allows to create a virtualenv and install the requirements.

```bash
make install
```

### Data

In order to ensure reproducibility, the data is separated in advance into two datasets: train (80%) and test (20%). Both datasets must be pulled from a remote repository, Google Drive in this case. To accomplish this task, [DVC](https://github.com/iterative/dvc) is used to allow data versioning. The following command will fetch the datasets to your local repository.

```bash
make data
```

> **_NOTE_**: The first time you try to download both datasets, DVC will ask you to authenticate with your Google account.

### Pipeline

In order to generate the pipeline, from preprocess data, perform hyperparameter optimization, generate the final model, store it in the models' registry and test its performance, the following command can be run.
```bash
make pipeline
```