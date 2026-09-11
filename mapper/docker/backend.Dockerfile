FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        colmap \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
COPY backend/requirements-automation.txt backend/requirements-automation.txt
RUN pip install --no-cache-dir -r backend/requirements-automation.txt

COPY backend /app/backend

EXPOSE 8081
CMD ["python", "backend/main.py"]

