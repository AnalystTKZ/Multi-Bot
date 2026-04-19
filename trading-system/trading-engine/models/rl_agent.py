"""
rl_agent.py — PPO-based strategy selector via Stable-Baselines3.

State: 42-dim vector.
Actions (v2 — expanded): 16
  0           = NoTrade
  1–5         = Trader 1–5 @ default threshold (0.55)
  6–10        = Trader 1–5 @ medium threshold (0.65)
  11–15       = Trader 1–5 @ high threshold (0.75)

The action encodes BOTH which strategy to trade AND how selective to be.
Higher-confidence thresholds mean fewer but better trades — the RL agent
can learn to dial selectivity up/down based on regime / drawdown context.

Backward-compat: select_action() still returns (trader_id, confidence_float)
so BaseTrader.analyze_market() is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

N_STATE = 42

# Action space layout
# [0]       NoTrade
# [1–5]     Traders 1–5 @ threshold tier 0 (0.55)
# [6–10]    Traders 1–5 @ threshold tier 1 (0.65)
# [11–15]   Traders 1–5 @ threshold tier 2 (0.75)
N_ACTIONS = 16

_THRESHOLD_TIERS = [0.55, 0.65, 0.75]
_N_TRADERS = 5
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # trading-engine/
MODEL_DIR = os.path.join(_MODEL_ROOT, "weights", "rl_ppo") + os.sep


# PPO with MLP policy is faster on CPU — GPU gives poor utilisation for small nets.
# SB3 explicitly warns against running MlpPolicy on GPU.
_RL_DEVICE = "cpu"
_BUFFER_TRIGGER = 64


def _decode_action(action_id: int) -> Tuple[int, float]:
    """
    Decode action_id → (trader_id [1–5 or 0], confidence_threshold).
    action 0      → (0, 0.0)   NoTrade
    action 1–5    → (1–5, 0.55)
    action 6–10   → (1–5, 0.65)
    action 11–15  → (1–5, 0.75)
    """
    if action_id == 0 or action_id >= N_ACTIONS:
        return (0, 0.0)
    tier = (action_id - 1) // _N_TRADERS          # 0, 1, or 2
    trader_num = ((action_id - 1) % _N_TRADERS) + 1  # 1–5
    threshold = _THRESHOLD_TIERS[min(tier, len(_THRESHOLD_TIERS)-1)]
    return (trader_num, threshold)


def _encode_action(trader_num: int, tier: int = 0) -> int:
    """Inverse of _decode_action."""
    if trader_num == 0:
        return 0
    return (tier * _N_TRADERS) + trader_num


class ModelNotTrainedError(RuntimeError):
    """Raised when the RL agent is used before training."""


class RLAgent(BaseModel):
    """
    PPO strategy selector. Heuristic fallback until model is trained.
    Interface contract preserved: record_outcome(), select_action().
    """

    weight_path = MODEL_DIR

    def __init__(self):
        super().__init__()
        self._model = None
        self._experience_buffer: Deque[dict] = deque(maxlen=2048)
        self._episode_count: int = 0
        self._rolling_rewards: Deque[float] = deque(maxlen=20)
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.is_trained:
            self.load(MODEL_DIR)

    @property
    def is_trained(self) -> bool:
        return os.path.exists(os.path.join(MODEL_DIR, "policy.pkl")) or \
               os.path.exists(os.path.join(MODEL_DIR, "model"))

    def record_outcome(self, trade_result: dict) -> None:
        """
        Called by TradeJournal after every trade close.
        Adds to experience buffer. Triggers mini-update when buffer >= 64.
        trade_result must contain: pnl, rr_ratio, confidence, rl_action, state_at_entry.
        """
        required = ("pnl", "rr_ratio", "confidence", "rl_action", "state_at_entry")
        if not all(k in trade_result for k in required):
            logger.debug("RLAgent.record_outcome: missing fields in trade_result")
            return

        reward = self._compute_reward(trade_result)
        self._rolling_rewards.append(reward)
        self._experience_buffer.append({
            "state": trade_result["state_at_entry"],
            "action": int(trade_result["rl_action"]),
            "reward": reward,
            "pnl": float(trade_result["pnl"]),
        })
        self._episode_count += 1

        if len(self._experience_buffer) >= _BUFFER_TRIGGER and self._model is not None:
            self._mini_update()

    def select_action(
        self, state: np.ndarray, available_signals: dict
    ) -> Tuple[int, float]:
        """
        Returns (trader_id [1–5 or 0], confidence_threshold).

        v2 action space: 16 actions.
          action 0       → NoTrade
          action 1–5     → Trader 1–5 @ threshold 0.55
          action 6–10    → Trader 1–5 @ threshold 0.65
          action 11–15   → Trader 1–5 @ threshold 0.75

        HARD RULE: if selected trader not in available_signals → return (0, 0.0).
        Raises ModelNotTrainedError if PPO weights are missing.
        Run: python scripts/retrain_incremental.py --model rl
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "RLAgent PPO model not trained. "
                "Run: python scripts/retrain_incremental.py --model rl"
            )

        self.reload_if_updated()

        obs = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=True)
        action_id = int(action[0]) if hasattr(action, "__len__") else int(action)
        action_id = max(0, min(action_id, N_ACTIONS - 1))

        trader_num, threshold = _decode_action(action_id)

        # Guard: strategy must have a rule-based signal
        if trader_num != 0:
            trader_key = f"trader_{trader_num}"
            if not available_signals.get(trader_key):
                return (0, 0.0)

        return (trader_num, threshold)

    def get_confidence_threshold(self, trader_num: int, state: np.ndarray) -> float:
        """
        Returns the dynamic confidence threshold for a trader given the current
        market state. Raises ModelNotTrainedError if model not loaded.
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "RLAgent PPO model not trained. "
                "Run: python scripts/retrain_incremental.py --model rl"
            )
        obs = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=True)
        action_id = int(action[0]) if hasattr(action, "__len__") else int(action)
        decoded_trader, threshold = _decode_action(action_id)
        if decoded_trader == trader_num:
            return threshold
        return _THRESHOLD_TIERS[0]

    def _compute_reward(self, trade_result: dict) -> float:
        """
        Multi-component reward:
          pnl_reward      = clip(r_multiple, -3, 4) × 1.0
          sharpe_bonus    = clip(rolling_sharpe_20 × 0.3, -0.5, 0.5)
          dd_penalty      = -2.0 × max(0, drawdown - 0.05)
          overtrade_pen   = -0.3 × max(0, trades_today - 4)
          session_bonus   = +0.15 if correct session strategy used AND profitable
          inaction_pen    = -0.05 if valid London setup skipped (action==0)
        """
        rr = float(trade_result.get("rr_ratio", 1.0))
        pnl = float(trade_result.get("pnl", 0.0))
        r_multiple = pnl / (abs(pnl / rr) + 1e-9) if rr > 0 else pnl
        pnl_reward = float(np.clip(r_multiple, -3, 4))

        rewards_list = list(self._rolling_rewards)
        if len(rewards_list) >= 2:
            mean_r = float(np.mean(rewards_list))
            std_r = float(np.std(rewards_list))
            sharpe_bonus = float(np.clip((mean_r / (std_r + 1e-9)) * 0.3, -0.5, 0.5))
        else:
            sharpe_bonus = 0.0

        drawdown = float(trade_result.get("drawdown_pct", 0.0))
        dd_penalty = -2.0 * max(0.0, drawdown - 0.05)

        trades_today = int(trade_result.get("trades_today", 0))
        overtrade_pen = -0.3 * max(0, trades_today - 4)

        rl_action = int(trade_result.get("rl_action", 0))
        session = str(trade_result.get("session", ""))
        session_bonus = 0.0
        if (session == "LONDON" and rl_action == 3) or \
           (session == "NY" and rl_action in (1, 2)) or \
           (session == "ASIAN" and rl_action in (4, 5)):
            if pnl > 0:
                session_bonus = 0.15

        inaction_pen = 0.0
        if rl_action == 0 and trade_result.get("missed_setup"):
            inaction_pen = -0.05

        total = pnl_reward + sharpe_bonus + dd_penalty + overtrade_pen + session_bonus + inaction_pen
        return float(np.clip(total, -5.0, 6.0))

    def _heuristic_fallback(
        self, available_signals: dict, session: str
    ) -> Tuple[int, float]:
        """
        When PPO not trained — returns (trader_id, confidence_threshold).
        Thresholds are conservative until model learns optimal values.
        """
        if session == "LONDON" and available_signals.get("trader_3"):
            return (3, 0.60)
        if session == "NY" and available_signals.get("trader_1"):
            return (1, 0.60)
        if session == "NY" and available_signals.get("trader_2"):
            return (2, 0.55)
        if session == "ASIAN" and available_signals.get("trader_5"):
            return (5, 0.55)
        if available_signals.get("trader_4"):
            return (4, 0.55)
        return (0, 0.0)

    def _detect_session(self) -> str:
        from datetime import datetime, timezone
        h = datetime.now(timezone.utc).hour
        if 2 <= h < 7:
            return "ASIAN"
        elif 7 <= h < 12:
            return "LONDON"
        elif 13 <= h < 18:
            return "NY"
        return "INACTIVE"

    def _mini_update(self) -> None:
        """Online PPO mini-update from experience buffer."""
        if self._model is None or len(self._experience_buffer) < _BUFFER_TRIGGER:
            return
        try:
            # In SB3 we can't easily do online updates from arbitrary buffers.
            # Log for deferred retraining via retrain_incremental.py.
            logger.debug("RLAgent: %d experiences buffered for next retrain", len(self._experience_buffer))
        except Exception as exc:
            logger.warning("RLAgent._mini_update failed: %s", exc)

    def retrain_from_journal(self, journal_path: str, n_epochs: int = 10) -> dict:
        """For retrain_incremental.py: reconstruct episodes from journal, run PPO update."""
        try:
            import gymnasium as gym  # noqa
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            episodes = self._load_journal_episodes(journal_path)
            if len(episodes) < 50:
                logger.warning("RLAgent.retrain: only %d episodes — skipping", len(episodes))
                return {"error": f"Only {len(episodes)} episodes — need >=50"}

            env = self._make_episode_env(episodes)
            _warm_start = (self._model is not None and self.is_trained)
            if not _warm_start:
                # Cold start: full LR, fresh policy network; ent_coef=0.01 encourages exploration
                self._model = PPO(
                    "MlpPolicy", env, verbose=0,
                    n_steps=64, batch_size=32,
                    n_epochs=n_epochs, learning_rate=3e-4,
                    ent_coef=0.01,
                    device=_RL_DEVICE,
                )
                logger.info("RLAgent: cold start — building new PPO policy")
            else:
                # Warm start: lower LR to fine-tune without forgetting prior policy
                self._model.set_env(env)
                self._model.learning_rate = 3e-4 / 5.0
                logger.info("RLAgent: warm start — fine-tuning existing PPO policy (lr=%.2e)", 3e-4 / 5.0)

            self._model.learn(total_timesteps=len(episodes) * n_epochs, reset_num_timesteps=False)
            self.save(MODEL_DIR)
            logger.info("RLAgent: retrain complete, %d episodes", len(episodes))
            return {"trained": True, "episodes": len(episodes)}
        except ImportError:
            logger.warning("RLAgent.retrain: stable-baselines3/gymnasium not available")
            return {"error": "stable-baselines3/gymnasium not available"}
        except Exception as exc:
            logger.error("RLAgent.retrain failed: %s", exc)
            return {"error": str(exc)}

    def _load_journal_episodes(self, journal_path: str) -> List[dict]:
        episodes = []
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "state_at_entry" in rec and len(rec["state_at_entry"]) == N_STATE:
                            episodes.append(rec)
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        return episodes

    def _make_episode_env(self, episodes: list):
        """Create a simple gym-like replay environment from journal episodes."""
        try:
            import gymnasium as gym
            from stable_baselines3.common.vec_env import DummyVecEnv

            class ReplayEnv(gym.Env):
                def __init__(self, eps):
                    super().__init__()
                    self.episodes = eps
                    self.idx = 0
                    self.observation_space = gym.spaces.Box(
                        low=-10, high=10, shape=(N_STATE,), dtype=np.float32
                    )
                    self.action_space = gym.spaces.Discrete(N_ACTIONS)  # 16 actions

                def reset(self, **kwargs):
                    self.idx = 0
                    obs = np.array(self.episodes[0]["state_at_entry"], dtype=np.float32)
                    return obs, {}

                def step(self, action):
                    ep = self.episodes[self.idx % len(self.episodes)]
                    obs = np.array(ep["state_at_entry"], dtype=np.float32)
                    pnl = float(ep.get("pnl", 0.0))
                    reward = float(np.clip(pnl / 100.0, -3, 4))
                    self.idx += 1
                    done = self.idx >= len(self.episodes)
                    return obs, reward, done, False, {}

            return DummyVecEnv([lambda: ReplayEnv(episodes)])
        except ImportError as exc:
            raise

    def save(self, path: str) -> None:
        if self._model is not None:
            try:
                os.makedirs(path, exist_ok=True)
                self._model.save(os.path.join(path, "model.zip"))
            except Exception as exc:
                logger.error("RLAgent.save failed: %s", exc)

    def load(self, path: str) -> None:
        try:
            from stable_baselines3 import PPO
            model_file = os.path.join(path, "model.zip")
            if os.path.exists(model_file):
                self._model = PPO.load(model_file, device=_RL_DEVICE)
                self._loaded = True
                logger.info("RLAgent: PPO model loaded from %s", model_file)
        except ImportError:
            logger.warning("RLAgent.load: stable-baselines3 not available")
        except Exception as exc:
            logger.error("RLAgent.load failed: %s", exc)
            self._model = None
