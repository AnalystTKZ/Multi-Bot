"""
rl_agent.py — PPO-based confidence-threshold selector via Stable-Baselines3.

State: 43-dim vector (regime logits, recent performance, market context).
Actions (v3 — threshold-only): 9
  0     = NoTrade
  1–8   = Trade with ml_trader at threshold [0.60, 0.62, 0.65, 0.68,
                                              0.70, 0.72, 0.75, 0.80]

The action encodes HOW SELECTIVE to be. Higher thresholds mean fewer but
higher-confidence trades. The RL agent learns to match selectivity to regime:
  - Strong trending + positive IC  → lower threshold (more trades)
  - Volatile ranging + degrading IC → higher threshold (fewer, better trades)
  - Drawdown / cooldown active      → NoTrade (action 0)

select_action() returns (trader_id=1, confidence_threshold) so the upstream
ml_trader interface is unchanged. trader_id=1 maps to the unified ml_trader.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel
from services.feature_engine import RL_STATE_DIM

logger = logging.getLogger(__name__)

N_STATE = RL_STATE_DIM
RL_STATE_SCHEMA_VERSION = "technical_unified_v2"

# Action space layout
# [0]     NoTrade
# [1]     Trade @ 0.60   (most permissive — use in strong trending + high IC)
# [2]     Trade @ 0.62
# [3]     Trade @ 0.65
# [4]     Trade @ 0.68
# [5]     Trade @ 0.70   (default baseline — matches settings.ML_DIRECTION_THRESHOLD)
# [6]     Trade @ 0.72
# [7]     Trade @ 0.75
# [8]     Trade @ 0.80   (most selective — use in volatile/ranging regimes)
N_ACTIONS = 9

_THRESHOLD_TIERS = [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # trading-engine/
MODEL_DIR = os.path.join(_MODEL_ROOT, "weights", "rl_ppo") + os.sep


# PPO with MLP policy is faster on CPU — GPU gives poor utilisation for small nets.
# SB3 explicitly warns against running MlpPolicy on GPU.
_RL_DEVICE = "cpu"
_BUFFER_TRIGGER = 64


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).lower() in ("1", "true", "yes", "on")


def _default_allowed_splits() -> set[str] | None:
    if _env_flag("ALLOW_NONTRAIN_JOURNAL_TRAINING"):
        return None
    default = "train,live,paper,production"
    if _env_flag("ALLOW_ROUND_JOURNAL_TRAINING", "0"):
        default = "train,validation,test,combined_eval,live,paper,production"
    raw = os.getenv("JOURNAL_ALLOWED_SPLITS", default)
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _record_allowed_by_split(record: dict, allowed_splits: set[str] | None) -> bool:
    if allowed_splits is None:
        return True
    split = str(record.get("source_split", "")).strip().lower()
    if split:
        return split in allowed_splits
    source = str(record.get("source", "")).strip().lower()
    if source.startswith("backtest_round_"):
        return False
    return _env_flag("ALLOW_UNTAGGED_JOURNAL_TRAINING")


def _decode_action(action_id: int) -> Tuple[int, float]:
    """
    Decode action_id → (trader_id, confidence_threshold).
    action 0   → (0, 0.0)  NoTrade
    action 1–8 → (1, _THRESHOLD_TIERS[action_id - 1])
    trader_id=1 maps to the unified ml_trader.
    """
    if action_id == 0 or action_id >= N_ACTIONS:
        return (0, 0.0)
    threshold = _THRESHOLD_TIERS[action_id - 1]
    return (1, threshold)


def _encode_action(threshold: float) -> int:
    """Return the action_id for the closest threshold tier."""
    if threshold <= 0:
        return 0
    diffs = [abs(t - threshold) for t in _THRESHOLD_TIERS]
    return diffs.index(min(diffs)) + 1


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
               os.path.exists(os.path.join(MODEL_DIR, "model")) or \
               os.path.exists(os.path.join(MODEL_DIR, "model.zip"))

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

    def decide(
        self, state: np.ndarray, available_signals: dict, session: str
    ) -> Tuple[int, float]:
        """
        Primary live-trading API. Returns (trader_id, confidence_threshold).
        Uses trained PPO when weights are present; session-aware heuristic before
        enough episodes have been collected to train the PPO.
        trader_id=0 means NoTrade regardless of source.
        """
        if self.is_trained and self._model is not None:
            return self.select_action(state, available_signals)
        logger.debug("RLAgent: untrained — heuristic fallback (session=%s)", session)
        return self._heuristic_fallback(available_signals, session)

    def select_action(
        self, state: np.ndarray, available_signals: dict
    ) -> Tuple[int, float]:
        """
        PPO inference path. Returns (trader_id=1, confidence_threshold) or (0, 0.0).
        Raises ModelNotTrainedError if PPO weights are missing — call decide() instead
        for the version that gracefully handles the untrained case.
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

        trader_id, threshold = _decode_action(action_id)

        if trader_id != 0 and not available_signals.get("ml_trader"):
            return (0, 0.0)

        return (trader_id, threshold)

    def get_confidence_threshold(self, state: np.ndarray) -> float:
        """
        Returns the dynamic confidence threshold for the current market state.
        Raises ModelNotTrainedError if model not loaded.
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "RLAgent PPO model not trained. "
                "Run: python scripts/retrain_incremental.py --model rl"
            )
        obs = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=True)
        action_id = int(action[0]) if hasattr(action, "__len__") else int(action)
        _, threshold = _decode_action(action_id)
        return threshold if threshold > 0 else _THRESHOLD_TIERS[0]

    def _compute_reward(self, trade_result: dict) -> float:
        """
        Multi-component reward (v3):
          pnl_reward    = clip(r_multiple, -3, 4)          — primary outcome
          sharpe_bonus  = clip(rolling_sharpe_20 × 0.3, -0.5, 0.5)
          dd_penalty    = -2.0 × max(0, drawdown - 0.05)  — drawdown deterrent
          overtrade_pen = -0.3 × max(0, trades_today - 3) — frequency cap
          session_bonus = +0.1 if London or NY session AND profitable
          inaction_pen  = -0.05 if valid setup skipped (action==0)
        """
        rr  = float(trade_result.get("rr_ratio", 1.0))
        pnl = float(trade_result.get("pnl", 0.0))
        r_multiple = pnl / (abs(pnl / rr) + 1e-9) if rr > 0 else pnl
        pnl_reward = float(np.clip(r_multiple, -3, 4))

        rewards_list = list(self._rolling_rewards)
        if len(rewards_list) >= 2:
            mean_r = float(np.mean(rewards_list))
            std_r  = float(np.std(rewards_list))
            sharpe_bonus = float(np.clip((mean_r / (std_r + 1e-9)) * 0.3, -0.5, 0.5))
        else:
            sharpe_bonus = 0.0

        drawdown     = float(trade_result.get("drawdown_pct", 0.0))
        dd_penalty   = -2.0 * max(0.0, drawdown - 0.05)

        trades_today  = int(trade_result.get("trades_today", 0))
        overtrade_pen = -0.3 * max(0, trades_today - 3)

        session       = str(trade_result.get("session", ""))
        session_bonus = 0.1 if (session in ("LONDON", "NY") and pnl > 0) else 0.0

        rl_action     = int(trade_result.get("rl_action", 0))
        inaction_pen  = -0.05 if (rl_action == 0 and trade_result.get("missed_setup")) else 0.0

        total = pnl_reward + sharpe_bonus + dd_penalty + overtrade_pen + session_bonus + inaction_pen
        return float(np.clip(total, -5.0, 6.0))

    def _heuristic_fallback(
        self, available_signals: dict, session: str
    ) -> Tuple[int, float]:
        """
        When PPO not trained — returns (1, threshold) or (0, 0.0).
        Session-aware threshold: London/NY get a slightly lower bar (more trades);
        Asian/Inactive get a higher bar (fewer, more conservative trades).
        """
        if not available_signals.get("ml_trader"):
            return (0, 0.0)
        if session in ("LONDON", "NY"):
            return (1, 0.65)
        if session == "ASIAN":
            return (1, 0.70)
        return (1, 0.72)

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

    def retrain_from_journal(
        self,
        journal_path: str,
        n_epochs: int = 10,
        allowed_splits: set[str] | None = None,
    ) -> dict:
        """For retrain_incremental.py: reconstruct episodes from journal, run PPO update."""
        try:
            import gymnasium as gym  # noqa
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            episodes = self._load_journal_episodes(journal_path, allowed_splits=allowed_splits)
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

    def _load_journal_episodes(
        self,
        journal_path: str,
        allowed_splits: set[str] | None = None,
    ) -> List[dict]:
        episodes = []
        allowed_splits = _default_allowed_splits() if allowed_splits is None else allowed_splits
        skipped_split = 0
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if not _record_allowed_by_split(rec, allowed_splits):
                            skipped_split += 1
                            continue
                        if "state_at_entry" in rec and len(rec["state_at_entry"]) == N_STATE:
                            episodes.append(rec)
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        if skipped_split:
            logger.info(
                "RLAgent: skipped %d journal records outside allowed splits %s",
                skipped_split,
                "ALL" if allowed_splits is None else sorted(allowed_splits),
            )
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
                    self.action_space = gym.spaces.Discrete(N_ACTIONS)  # 9 actions

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
                with open(os.path.join(path, "state_schema.json"), "w") as f:
                    json.dump(
                        {
                            "version": RL_STATE_SCHEMA_VERSION,
                            "n_state": N_STATE,
                        },
                        f,
                        indent=2,
                    )
            except Exception as exc:
                logger.error("RLAgent.save failed: %s", exc)

    def load(self, path: str) -> None:
        try:
            from stable_baselines3 import PPO
            model_file = os.path.join(path, "model.zip")
            if os.path.exists(model_file):
                schema_file = os.path.join(path, "state_schema.json")
                if not os.path.exists(schema_file):
                    logger.warning(
                        "RLAgent: refusing to load %s because state_schema.json is missing; retrain RL.",
                        model_file,
                    )
                    return
                with open(schema_file) as f:
                    schema = json.load(f)
                if (
                    schema.get("version") != RL_STATE_SCHEMA_VERSION
                    or int(schema.get("n_state", -1)) != N_STATE
                ):
                    logger.warning(
                        "RLAgent: refusing stale PPO state schema %s; expected version=%s n_state=%d.",
                        schema,
                        RL_STATE_SCHEMA_VERSION,
                        N_STATE,
                    )
                    return
                self._model = PPO.load(model_file, device=_RL_DEVICE)
                self._loaded = True
                logger.info("RLAgent: PPO model loaded from %s", model_file)
        except ImportError:
            logger.warning("RLAgent.load: stable-baselines3 not available")
        except Exception as exc:
            logger.error("RLAgent.load failed: %s", exc)
            self._model = None
