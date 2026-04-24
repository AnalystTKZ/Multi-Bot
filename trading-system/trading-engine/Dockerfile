FROM python:3.11-slim

# Build essentials for LightGBM, XGBoost, pandas, faiss-cpu, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU-only wheel.
# On ARM64 (Oracle Cloud A1/Ampere) the pytorch.org/whl/cpu index has no arm64 wheels —
# PyPI itself carries the correct aarch64 CPU wheel, so we omit --index-url on ARM.
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch; \
    fi

# Install everything else
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

COPY . .

# Ensure runtime directories exist
RUN mkdir -p logs weights/gru_lstm weights/rl_ppo weights/backups

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]
