# Reproducible environment for training and testing the failure predictor.
#
#   docker build -t failure-predictor .
#   docker run --rm failure-predictor                  # train + evaluate (default)
#   docker run --rm failure-predictor pytest -q        # run the test suite
#   docker run --rm -v "$(pwd)/results:/app/results" failure-predictor   # keep results on the host

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install pinned dependencies first, so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fp/ fp/
COPY tests/ tests/
COPY data/ data/
COPY train.py pytest.ini ./

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 appuser && mkdir -p results && chown -R appuser:appuser /app
USER appuser

CMD ["python", "train.py"]
