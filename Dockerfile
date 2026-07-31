ARG PYTHON_IMAGE=python:3.13.6-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /workspace
COPY requirements.lock pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY clinical_validation/config ./clinical_validation/config
COPY results ./results
COPY schemas ./schemas
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
    && python -m pip install --no-cache-dir -r requirements.lock \
    && python -m pip install --no-cache-dir --no-deps . \
    && tm-ecg-verify-reported

ENTRYPOINT ["tm-ecg"]
CMD ["doctor"]
