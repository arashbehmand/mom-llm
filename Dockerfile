# MoM v2 image — uv-based, multi-stage, non-root. Built from the repo root
# (`docker build -f docker/Dockerfile .`). The dependency layer is cached from the lockfile.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependency layer (cached until pyproject/uv.lock change).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Project layer.
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.13-slim AS runtime

RUN useradd --create-home --uid 10001 app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    MOM_DATA_DIR=/data \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data && chown app:app /data
VOLUME /data
EXPOSE 8000
USER app

# Healthcheck uses the CLI (stdlib urllib) — no undeclared dependency.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["mom", "healthcheck"]

ENTRYPOINT ["mom"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
