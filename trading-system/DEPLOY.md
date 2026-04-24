# Oracle Cloud A1 Deployment Guide (ARM64 Ampere — Always Free)

## 1. Create the Oracle Cloud VM

1. Sign up at cloud.oracle.com → create a free account
2. Go to **Compute → Instances → Create Instance**
3. Change image to **Ubuntu 22.04** (Canonical)
4. Change shape to **Ampere → VM.Standard.A1.Flex**
   - Set **4 OCPUs** and **24 GB RAM** (maximum free tier)
5. Download the SSH key pair when prompted — save it as `oracle_key.pem`
6. Click Create

Once running, note the **Public IP address**.

---

## 2. Open Firewall Ports

In Oracle Cloud Console → **Virtual Cloud Network → Security Lists → Add Ingress Rules**:

| Port | Protocol | Purpose |
|------|----------|---------|
| 22   | TCP      | SSH |
| 3000 | TCP      | Backend API |
| 3001 | TCP      | Frontend dashboard |
| 8000 | TCP      | Trading engine (internal — optional to expose) |

Also run on the VM itself (Ubuntu ufw):
```bash
sudo ufw allow 22 && sudo ufw allow 3000 && sudo ufw allow 3001 && sudo ufw enable
```

---

## 3. Install Docker on the VM

SSH in first:
```bash
ssh -i oracle_key.pem ubuntu@YOUR_SERVER_IP
```

Then install Docker:
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
newgrp docker
docker --version   # verify
```

Install Docker Compose plugin:
```bash
sudo apt-get install -y docker-compose-plugin
docker compose version   # verify
```

---

## 4. Clone the Repo

```bash
git clone https://github.com/AnalystTKZ/Multi-Bot.git
cd Multi-Bot/trading-system
```

---

## 5. Set Up Environment

Copy `.env.production` from your local machine to the server:
```bash
# Run this on your LOCAL machine:
scp -i oracle_key.pem trading-system/.env.production ubuntu@YOUR_SERVER_IP:~/Multi-Bot/trading-system/.env
```

Then on the server, fill in all `CHANGE_ME_` values:
```bash
nano ~/Multi-Bot/trading-system/.env
```

Values you must set:
- `DB_PASSWORD` — strong random password
- `REDIS_PASSWORD` — strong random password
- `JWT_SECRET` — run: `python3 -c "import secrets; print(secrets.token_hex(32))"`
- `ADMIN_PASSWORD` — your dashboard login password
- `CAPITAL_API_KEY`, `CAPITAL_IDENTIFIER`, `CAPITAL_PASSWORD` — from Capital.com account
- `VITE_API_URL` / `VITE_WS_URL` — replace `YOUR_SERVER_IP` with real VM IP

---

## 6. Copy Trained Model Weights

The weights are already tracked in git, so `git clone` includes them.
Verify they exist after cloning:
```bash
ls ~/Multi-Bot/trading-system/trading-engine/weights/
# Expected: gru_lstm/  rl_ppo/  regime_4h.pkl  regime_1h.pkl  quality_scorer.pkl
```

If any are missing, copy from local:
```bash
# Run on LOCAL machine:
scp -i oracle_key.pem -r trading-system/trading-engine/weights/ \
    ubuntu@YOUR_SERVER_IP:~/Multi-Bot/trading-system/trading-engine/
```

---

## 7. Build and Start

```bash
cd ~/Multi-Bot/trading-system

# First build (takes ~10-15 min on ARM — PyTorch download is large)
docker compose -f docker-compose.dev.yml build

# Start all services
docker compose -f docker-compose.dev.yml up -d

# Watch startup logs
docker compose -f docker-compose.dev.yml logs -f
```

---

## 8. Verify Everything is Running

```bash
# All 6 containers should show "Up"
docker compose -f docker-compose.dev.yml ps

# Backend health
curl http://localhost:3000/health

# Trading engine health
curl http://localhost:8000/health

# Frontend reachable
curl -I http://localhost:3001
```

Dashboard URL: `http://YOUR_SERVER_IP:3001`
API URL: `http://YOUR_SERVER_IP:3000/api`

---

## 9. Keeping the Bot Running After Reboot

Docker's `restart: unless-stopped` handles this automatically for:
- `trading-engine`
- `model-retrainer`

To also start the full stack on reboot:
```bash
# Enable Docker to start on boot
sudo systemctl enable docker

# Optionally add a cron job to bring compose up on reboot
(crontab -l 2>/dev/null; echo "@reboot cd ~/Multi-Bot/trading-system && docker compose -f docker-compose.dev.yml up -d") | crontab -
```

---

## 10. Useful Commands

```bash
# View live logs for trading engine
docker logs -f trading_engine_main

# View live logs for retrainer
docker logs -f trading_model_retrainer

# Restart a single service
docker compose -f docker-compose.dev.yml restart trading-engine

# Stop everything
docker compose -f docker-compose.dev.yml down

# Update code and rebuild
git pull
docker compose -f docker-compose.dev.yml build trading-engine
docker compose -f docker-compose.dev.yml up -d trading-engine
```

---

## 11. Switch to Live Trading

When ready to go live (real money):
1. Edit `.env` on the server:
   ```
   PAPER_TRADING=false
   CAPITAL_ENV=live
   CAPITAL_API_KEY=your_live_api_key
   ```
2. Restart the trading engine:
   ```bash
   docker compose -f docker-compose.dev.yml restart trading-engine
   ```

---

## Resource Usage (expected on A1 4 OCPU / 24 GB RAM)

| Service | RAM | CPU |
|---------|-----|-----|
| postgres | ~256 MB | low |
| redis | ~64 MB | low |
| backend | ~256 MB | low |
| trading-engine | ~1.5-2 GB | medium (inference) |
| frontend (nginx) | ~32 MB | very low |
| model-retrainer | ~1 GB (weekly peak) | medium |
| **Total headroom** | **~20 GB free** | comfortable |
