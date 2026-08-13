FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir array-api-compat numpy scipy matplotlib
RUN pip install --no-cache-dir "jax[cuda12]" cupy-cuda12x

WORKDIR /workspace
ENTRYPOINT ["python3"]