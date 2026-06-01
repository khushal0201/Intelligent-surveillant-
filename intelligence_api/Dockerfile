FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/intelligence_api
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "intelligence_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
