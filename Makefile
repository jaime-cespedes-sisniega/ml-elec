.ONESHELL:
SHELL := /bin/bash

VENV=.venv

install:
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	pip3 install --upgrade pip &&\
				 pip3 install -r requirements/requirements.txt \
				              -r requirements/tox_requirements.txt

train:
	$(VENV)/bin/python train_pipeline.py

clean_data:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv

clean_models:
	rm -f model/*.joblib

clean_virtualenv:
	rm -rf $(VENV)

clean_all: clean_data clean_models clean_virtualenv
