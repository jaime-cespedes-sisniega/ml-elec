# ML Pipeline
This repo aims to generate a simple Machine Learning pipeline using an open dataset with data collected from the [Australian Electricity Market](https://www.openml.org/d/151).

## Reproducibility

In order to facilitate the reproducibility of the pipeline, raw data can be obtained from https://www.openml.org/d/151 using `train_pipeline.py`. It is responsible for downloading the data from the original source, and splitting the data set into training and test sets maintaining temporal sequentiality with the objective of using the test dataset to detect changes in the data distributions (Data Drift) or in the behavior of the independent variables with respect to the dependent variable (Concept Drift). In addition, the script generates a simple model (Random Forest), which will be used to generate the necessary predictions in order to evaluate the presence of some type of drift.

## API

### Build

```bash
make build-api
```

### Usage

```bash
make run-api
```
