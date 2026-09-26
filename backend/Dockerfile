# Backend image — FastAPI app served by uvicorn.
#
# IMPORTANT: build with the REPO ROOT as the context (see docker-compose.yml:
#   build: { context: ., dockerfile: backend/Dockerfile }
# ). The pipeline resolves the dataset from <repo-root>/data (the retriever and
# eval harness walk up the source tree to find it), so the image must mirror the
# repo layout: /app/backend/... alongside /app/data/....
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LAYER ORDER IS LOAD-BEARING. Docker invalidates a layer and EVERYTHING below
# it when its inputs change, so copying source before the install made a
# one-character code edit re-download and reinstall every dependency (~10 min a
# rebuild). The manifest is copied alone first, so the install layer below is
# invalidated only when pyproject.toml itself changes; source and data land
# after it.
COPY backend/pyproject.toml ./backend/
WORKDIR /app/backend

# CPU-ONLY TORCH, installed BEFORE the project so `pip install -e .` finds the
# requirement already satisfied and never resolves the CUDA variant.
#
# `sentence-transformers` is the only torch consumer, and torch's own metadata
# requires five nvidia-* packages gated `platform_system == "Linux"` (cublas,
# cudnn, cusparselt, nccl, nvshmem) which transitively pull cusolver/curand/
# cufft — roughly 3 GB of CUDA runtime this image can never use. That Linux
# marker is why the Windows dev venv installs ZERO nvidia packages while the
# container installs all of them.
#
# Nothing here can use a GPU: every SentenceTransformer(...) is constructed with
# an explicit device="cpu" (faiss_retriever, trainable_classifier, bench_real),
# faiss is pinned CPU-only, and no .cuda()/torch.device call exists in app/.
#
# Version PINNED to what the default index currently resolves, so this changes
# WHERE torch comes from and not WHICH torch — a CPU switch and a version bump
# in one step would make any behaviour change ambiguous.
#
# Dockerfile-only on purpose: pyproject.toml stays portable, so the Windows dev
# venv is unaffected. The stub package exists because `[tool.setuptools]
# packages = ["app"]` needs it at install time; the real tree overwrites it
# below, and the editable path hook resolves at import time.
RUN mkdir -p app && touch app/__init__.py \
 && pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 \
 && pip install -e .

# Source and the shared dataset (kept at repo root by design) last: a code
# change now invalidates only these layers, not the install above.
WORKDIR /app
COPY backend/ ./backend/
COPY data/ ./data/
WORKDIR /app/backend

EXPOSE 8000

# Apply migrations, then serve. Shell form so both run in sequence.
# alembic and uvicorn both read DATABASE_URL (the async URL — migrations/env.py
# injects it into Alembic's sqlalchemy.url); uvicorn serves on 0.0.0.0.
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]
