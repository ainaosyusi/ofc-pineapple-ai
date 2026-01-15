# OFC Pineapple AI - Training Environment
# Multi-stage build for optimized image size

FROM python:3.9-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and build C++ extension
COPY setup.py .
COPY src/cpp/ src/cpp/
RUN python setup.py build_ext --inplace

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy built artifacts
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/*.so /app/
COPY --from=builder /app/build /app/build

# Copy source code
COPY src/ src/
COPY setup.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default training parameters
ENV TOTAL_TIMESTEPS=5000000
ENV OPPONENT_UPDATE_FREQ=50000
ENV NOTIFICATION_INTERVAL=100000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import ofc_engine; print('OK')" || exit 1

# Run training
CMD ["sh", "-c", "python src/python/train_phase1.py \
    --timesteps ${TOTAL_TIMESTEPS} \
    --lr 0.0003"]
