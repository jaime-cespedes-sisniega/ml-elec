# ML Pipeline
This repo aims to generate a simple Machine Learning pipeline using an open dataset with data collected from the [Australian Electricity Market](https://www.openml.org/d/151).

## Reproducibility

In order to facilitate the reproducibility of the pipeline, raw data can be acquired from https://www.openml.org/d/151 using `train_pipeline.py` (`data/raw` folder is empty) right before the training is carried out. Then a preprocessing step takes place, in which raw data is split in train and test data. With the intention of maintaining the temporal order of the data samples, test data is not shuffle. 

## API

### Build

```bash
make build-api
```

### Usage

```bash
make run-api
```
