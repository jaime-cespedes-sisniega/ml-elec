.ONESHELL:
SHELL := /bin/bash

VENV=.venv

install:
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	pip3 install --upgrade pip &&\
				 pip3 install -r requirements/requirements.txt \
				              -r requirements/tox_requirements.txt

pipeline:
	$(VENV)/bin/python run_pipeline.py

clean_virtualenv:
	rm -rf $(VENV)

clean_data:
	rm data/*.csv

clean: clean_data clean_virtualenv