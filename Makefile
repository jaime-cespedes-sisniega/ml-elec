train:
	python train_pipeline.py

clean_data:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv

clean_models:
	rm -f model/*.joblib

clean_all: clean_data clean_models
