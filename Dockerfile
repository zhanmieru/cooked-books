# CookedBooks verification image. Built and run by `make verify`.
#
# The build needs no network once the base image is present locally: test
# dependencies come only from vendor/wheels (pure-Python wheels, so the
# image builds the same on amd64 and arm64). The base image is pinned by
# tag and by the digest of its multi-arch index.
FROM python:3.11.18-slim-trixie@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_INDEX=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis \
    LANG=C.UTF-8 \
    TZ=UTC

WORKDIR /app

# Dependencies first, so a source change does not reinstall them.
COPY requirements.lock ./
COPY vendor/wheels/ ./vendor/wheels/
RUN python -m pip install --no-index --only-binary=:all: \
        --find-links ./vendor/wheels -r requirements.lock \
 && groupadd --system --gid 10001 cookedbooks \
 && useradd --system --uid 10001 --gid cookedbooks --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin cookedbooks

# Source last. Files stay owned by root: the run user cannot change them.
COPY . .

USER 10001:10001
CMD ["python", "-m", "cookedbooks.verify"]
