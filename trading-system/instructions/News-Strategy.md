# News Momentum Strategy — Trader 4

Updated 2026-04-02. **SUPERSEDED 2026-04-19.**

> `trader_4_news_momentum.py` has been deleted. News sentiment is now encoded as
> `sentiment_score` (QUALITY_FEATURES index 6) fed to the QualityScorer, and as news
> proximity flags (`news_in_30min`, `news_in_15min`) in the feature pipeline.
> This document is retained as historical context only.

Implementation: `trading-engine/traders/trader_4_news_momentum.py`

---

## Overview

Trades structural breakouts triggered by high-impact economic events. Unlike a news fade, this strategy rides the **continuation** of an impulse once sentiment and structure confirm direction. The FinBERT sentiment gate is mandatory — no trade fires without it.

**Active pairs**: All configured trading pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, XAUUSD). Gold is explicitly included — macro events like CPI, FOMC, and NFP directly drive XAU/USD.

---

## Strategy Type: Structural Breakout (NOT a Fade)

The previous Trader 4 was a post-spike fade (enter opposite the news impulse). The current implementation is the opposite: it **follows** the sentiment direction via a structural breakout. This is a fundamental design difference:

| Old T4 (news fade) | New T4 (news momentum) |
|---|---|
| Enter against the initial spike | Enter in the direction of the spike |
| Waits 3–30 min for reversal | Enters on structural confirmation of continuation |
| Needs spike exhaustion signals | Needs FinBERT sentiment ≥ 0.65 + BOS in sentiment direction |
| VADER + finance lexicon | FinBERT primary (ProsusAI/finbert), VADER fallback |

---

## Tracked High-Impact Events

| Event | Why It Creates Momentum |
|---|---|
| NFP (Non-Farm Payrolls) | Largest monthly USD directional event |
| CPI / PPI | Inflation data drives rate expectations and sustained moves |
| FOMC statements / rate decisions | Policy decisions create durable directional impulses |
| GDP | Large surprises commit institutional flow for hours |
| Retail Sales | Consumer demand proxy with persistent post-release trend |
| Major central bank speeches | Clear hawkish/dovish guidance drives sustained positioning |

---

## Signal Logic

### Gate 1 — News Presence (HARD gate)
`news_in_15min = True` — a high-impact event must be within the 15-minute window.
Source: `NewsService.get_upcoming_events()` polling ForexFactory.

### Gate 2 — Sentiment Score (HARD gate)
`sentiment_score ≥ 0.65` — FinBERT (or VADER fallback) must produce strong directional sentiment.
No trade fires if this threshold is not met, regardless of structure.

For XAUUSD: USD-negative FinBERT sentiment is **inverted** to XAU-positive (weak USD = strong Gold).

### Gate 3 — Structural Breakout
Price must break a recent swing high (bullish sentiment) or swing low (bearish sentiment) — i.e., Break of Structure (BOS) in the sentiment direction. This confirms that institutional flow is committing, not just reacting.

---

## Risk Management

- **SL**: beyond the breakout candle low (long) or high (short)
- **TP**: minimum 1.5× ATR from entry; extended to next structural target when available
- **Hard close**: 60 minutes after the triggering news event — no position held into the next session
- **Max 1 trade per session** — one clean read per event, no second attempts
- `SENTIMENT_THRESHOLD = 0.65` (class constant, not configurable via .env)

---

## Bot Implementation

| Parameter | Value |
|---|---|
| TRADER_ID | `trader_4` |
| Session | Any (no session restriction) |
| Sentiment gate | FinBERT ≥ 0.65 (HARD) |
| News window | `news_in_15min = True` required |
| Structure | BOS in sentiment direction required |
| Entry | Market order in sentiment direction after BOS |
| Stop | Beyond breakout candle extreme |
| TP | 1.5× ATR minimum, to next structure |
| Max trades/session | 1 |
| Hard close | 60 min post-news |
| File | `trading-engine/traders/trader_4_news_momentum.py` |

---

## Sentiment Model Details

**Primary**: FinBERT (`ProsusAI/finbert`) — finance-domain BERT model
- Input: news headline string
- Output: positive, negative, neutral class probabilities
- `sentiment_score = positive_prob − negative_prob` (range: −1 to +1)

**Fallback**: VADER — if transformers unavailable or inference fails
- VADER compound score normalised to same −1 to +1 range

**Gold inversion**: when symbol is `XAUUSD`, a negative USD sentiment score is flipped positive (USD weakness → Gold strength). This is applied inside `SentimentModel.analyze()`.

**Caching**: LRU cache with 50 entries, TTL 30 minutes — avoids repeated inference on the same headline within a session.

---

## Operational Notes

- `NewsService` (`services/news_service.py`) polls ForexFactory on a background thread. `is_blocked()` returns True within 15 min of a high-impact event for suppression in other traders. `get_upcoming_events()` returns the event list that Trader 4 uses for its gate.
- If `ML_ENABLED=false`, FinBERT/VADER still loads because `SentimentModel` uses pre-trained weights (no training required). It is the only model not gated by `ML_ENABLED`.
- `NEWS_API_KEY` in `.env` enables additional news sources beyond ForexFactory scraping.
