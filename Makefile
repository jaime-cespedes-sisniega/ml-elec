clean_data:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv

clean_models:
	rm -f models/*.joblib

clean_all: clean_data clean_models

serve-api-dev:
	uvicorn api.main:api --host 0.0.0.0 --port 5000 --workers 5  # Num workers

serve-prod:
	gunicorn -b 0.0.0.0:5000 -w 5 -k uvicorn.workers.UvicornWorker api.main:api # Num workers = (2*CPU) + 1