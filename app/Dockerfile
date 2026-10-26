# Build context is the repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-vie \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY app/pyproject.toml app/pyproject.toml
# Dependencies come from pyproject.toml; installing them before copying the code keeps this
# layer cached across code changes.
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('app/pyproject.toml','rb'))['project']['dependencies']))" > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY app app
# data/ is bind-mounted at runtime (docker-compose.yml), not baked into the image.
EXPOSE 8080
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
