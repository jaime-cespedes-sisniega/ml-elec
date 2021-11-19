FROM python:3.8.11-slim

EXPOSE 5000

COPY api/requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY ./ml_pipeline /ml_pipeline
COPY setup.py .
RUN pip install . --use-feature=in-tree-build --no-cache-dir

COPY ./api /api

# Default to 1 worker
ENV num_workers 1

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:5000 -w ${num_workers} -k uvicorn.workers.UvicornWorker api.main:api"]