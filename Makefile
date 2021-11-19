train:
	python train_pipeline.py

clean_data:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv

clean_models:
	rm -f api/models/*.joblib

clean_all: clean_data clean_models

serve-api-dev:
	uvicorn api.main:api --host 0.0.0.0 --port 5000 --workers 5  # Num workers

serve-api-prod:
	gunicorn -b 0.0.0.0:5000 -w 5 -k uvicorn.workers.UvicornWorker api.main:api # Num workers = (2*CPU) + 1

build-api:
	docker build -t ml-api .

run-api:
	docker run -d --name ml-api -p 5000:5000 -e num_workers=5 ml-api