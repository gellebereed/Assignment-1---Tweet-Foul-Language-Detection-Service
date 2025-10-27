FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY service /app/service
COPY artifacts /app/artifacts

ENV MODEL_ARTIFACT=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
