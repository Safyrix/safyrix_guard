FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 appuser

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import joblib; print('joblib OK', joblib.__version__)"


COPY app /app/app
COPY models /app/models

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
