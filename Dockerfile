# Multi-stage build for smaller image size
# Stage 1: Build dependencies
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsqlite3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies (install globally so binaries are available system-wide)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder (system-wide install under /usr/local)
COPY --from=builder /usr/local /usr/local

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser ./mom_service ./mom_service
COPY --chown=appuser:appuser ./docker/start-uvicorn.sh /usr/local/bin/start-uvicorn.sh
# Note: config.yaml should be mounted as a volume or provided via environment

# Create directories for databases with correct permissions
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data
RUN chmod +x /usr/local/bin/start-uvicorn.sh

# Set environment variables
ENV PYTHONPATH=/app \
    PATH=/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

ENTRYPOINT ["/usr/local/bin/start-uvicorn.sh"]
