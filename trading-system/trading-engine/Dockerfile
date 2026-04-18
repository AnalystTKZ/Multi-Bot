FROM python:3.11-slim

# Build essentials for LightGBM, XGBoost, pandas, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU-only wheel first (~700MB vs ~2.5GB for the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

COPY . .

# Ensure runtime directories exist
RUN mkdir -p logs weights/gru_lstm weights/rl_ppo weights/backups

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]
