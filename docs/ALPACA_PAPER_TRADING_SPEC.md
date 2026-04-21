# Alpaca Paper Trading Integration Specification
## For Claude Code Implementation

---

## Overview

Integrate Alpha Scanner's scoring system with Alpaca's Paper Trading API to automatically execute trades based on breakout signals. The system should run daily after market close, score all tickers, and place buy/sell orders through Alpaca's paper trading environment.

This is paper trading only — no real money. The goal is to validate the strategy in a live-market simulation for 1+ months before considering real capital.

---

## Prerequisites

Todd has already:
- Created an Alpaca account at https://app.alpaca.markets
- Switched to Paper Trading mode
- Generated API Key and Secret

The keys need to be stored as environment variables:
```bash
export ALPACA_API_KEY="your-key-here"
export ALPACA_SECRET_KEY="your-secret-here"
```

---

## Python SDK

Use the official `alpaca-py` SDK (NOT the deprecated `alpaca-trade-api-python`):

```bash
pip install alpaca-py
```

Key imports:
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
```

Initialize with paper=True:
```python
trading_client = TradingClient(api_key, secret_key, paper=True)
```

---

## Trading Rules (from backtested portfolio simulation)

### Entry Rules
- **Threshold:** Score ≥ 9.5
- **No persistence filter** (backtesting showed it hurts at 9.5+ by delaying fast breakout entries)
- **Position size:** Max 20% of portfolio equity per position (Alpaca paper starts with $100k)
- **Order type:** Market order at next day's open
- **No re-entry within 30 days of a loss exit** (wash sale rule)

### Exit Rules (whichever triggers first)
1. **Score exit:** Ticker's score drops below 5 → sell at next day's open
2. **Stop loss:** Price drops 15% below entry price → sell at next day's open

### Capital Rotation
- **Strategy: No rotation** (simplest approach validated in backtest at +356.6%)
- If no cash is available when a new signal appears, skip it
- Capital frees up naturally as positions exit via score or stop loss

### Excluded Tickers
- Crypto tickers (BTC-USD, ETH-USD, etc.) — Alpaca handles crypto differently, exclude for now
- Futures tickers (GC=F, SI=F) — not tradeable on Alpaca
- Any ticker that Alpaca doesn't recognize as tradeable (check with `trading_client.get_asset()`)

---

## Architecture

### File: `trade_executor.py`

This is the main script that runs daily. It should:

1. **Fetch current portfolio state from Alpaca**
   - Cash available
   - Open positions (ticker, qty, entry price, current price, unrealized P&L)

2. **Score all tickers** using the existing Alpha Scanner pipeline
   - `load_config()` → `fetch_all()` → `score_all()`

3. **Check exit conditions on existing positions**
   - For each open Alpaca position:
     - Get the ticker's current score
     - Get the ticker's entry price (from Alpaca position data)
     - If score < 5: queue a sell order
     - If current price ≤ entry_price × 0.85: queue a sell order (15% stop)
   - Execute all sell orders

4. **Check entry conditions for new positions**
   - Filter scored tickers for score ≥ 9.5
   - Exclude tickers already in portfolio
   - Exclude tickers in the wash sale cooldown window (sold at a loss within last 30 days)
   - Exclude crypto and futures tickers
   - Verify each ticker is tradeable on Alpaca
   - Sort remaining by score descending
   - For each candidate:
     - Calculate position size: min(available_cash, portfolio_equity × 0.20)
     - If position size ≥ $500 (minimum threshold): place a market buy order
     - Update available cash

5. **Log everything**
   - Every decision (buy, sell, skip, wash sale block) with reasoning
   - Portfolio summary after all trades

### File: `wash_sale_tracker.py`

Tracks loss exits to enforce the 30-day wash sale window.

```python
# Simple JSON file persistence
# wash_sale_log.json
{
    "VIAV": {
        "exit_date": "2026-03-09",
        "loss_amount": -4200.50,
        "cooldown_until": "2026-04-08"
    }
}
```

Functions needed:
- `record_loss_exit(ticker, exit_date, loss_amount)` — adds to log
- `is_blocked(ticker, current_date)` — returns True if within 30-day window
- `get_blocked_tickers(current_date)` — returns list of all blocked tickers
- `cleanup_expired(current_date)` — removes entries older than 30 days

### File: `trade_log.py`

Maintains a persistent trade log (separate from Alpaca's records) for our own analysis.

```python
# trade_history.json or trade_history.csv
{
    "trades": [
        {
            "ticker": "AAOI",
            "side": "buy",
            "date": "2026-04-10",
            "price": 45.20,
            "qty": 442,
            "cost_basis": 19978.40,
            "score_at_entry": 9.8,
            "reason": "Score ≥ 9.5"
        },
        {
            "ticker": "AAOI",
            "side": "sell",
            "date": "2026-05-15",
            "price": 38.42,
            "qty": 442,
            "proceeds": 16981.64,
            "pnl": -2996.76,
            "pnl_pct": -15.0,
            "score_at_exit": 6.2,
            "reason": "Stop loss (15%)",
            "hold_days": 35
        }
    ]
}
```

---

## Daily Run Flow

```
trade_executor.py (run daily after 5 PM ET)
│
├── 1. Connect to Alpaca (paper=True)
│   └── Get account info: equity, cash, buying power
│
├── 2. Score all tickers
│   ├── load_config()
│   ├── fetch_all(cfg, period="1y")
│   └── score_all(data, cfg)
│
├── 3. Process EXITS (before entries)
│   ├── Get all open positions from Alpaca
│   ├── For each position:
│   │   ├── Find ticker's current score
│   │   ├── Check if score < 5 → SELL
│   │   ├── Check if price ≤ entry × 0.85 → SELL (stop loss)
│   │   └── If selling at a loss → record in wash_sale_tracker
│   └── Submit all sell orders
│
├── 4. Process ENTRIES (after exits settle)
│   ├── Get updated cash balance
│   ├── Filter: score ≥ 9.5, not already held, not wash-sale blocked
│   ├── Verify tradeable on Alpaca
│   ├── Sort by score descending
│   ├── For each candidate:
│   │   ├── Calculate position size (max 20% of equity)
│   │   ├── Submit market buy order
│   │   └── Log trade
│   └── Stop when cash runs out
│
├── 5. Generate daily summary
│   ├── Portfolio value, cash, positions held
│   ├── Trades executed today (buys + sells)
│   ├── Blocked tickers (wash sale)
│   ├── Signals skipped (no cash)
│   └── Print to console + append to daily_log.txt
│
└── 6. (Optional) Send email digest with trade activity
```

---

## Console Output Format

```
================================================================
  ALPHA SCANNER — DAILY TRADE EXECUTION
  2026-04-10 17:30 PT
================================================================

  ACCOUNT STATUS
  ──────────────────────────────────────
  Equity:          $108,450.20
  Cash:            $22,300.50
  Positions:       6
  Day's P&L:       +$1,240.30

  EXITS TODAY
  ──────────────────────────────────────
  SELL  CRWD    442 shares @ $285.30    Score: 4.2 (score exit)     P&L: +$3,200 (+8.1%)
  SELL  WOLF    890 shares @ $12.40     Stop loss hit (-15%)        P&L: -$2,650 (-15.0%)
                                        → Wash sale recorded: blocked until 2026-05-10

  ENTRIES TODAY
  ──────────────────────────────────────
  BUY   AAOI    520 shares @ $38.50     Score: 9.8    Cost: $20,020
  BUY   CIEN    180 shares @ $98.20     Score: 9.5    Cost: $17,676

  SKIPPED SIGNALS
  ──────────────────────────────────────
  SKIP  VIAV    Score: 9.7    Reason: Wash sale block (until 2026-04-25)
  SKIP  LUNR    Score: 9.5    Reason: Insufficient cash ($4,624 available, need $20,000)

  PORTFOLIO POSITIONS
  ──────────────────────────────────────
  Ticker    Qty    Entry     Current    P&L        Score    Hold Days
  AAOI      520    $38.50    $38.50     +0.0%      9.8      0
  CIEN      180    $98.20    $98.20     +0.0%      9.5      0
  LITE      310    $55.40    $72.10     +30.1%     7.2      28
  NEM       220    $88.60    $95.30     +7.6%      6.8      14
  SLV       340    $42.80    $44.10     +3.0%      6.1      21
  KTOS      160    $32.50    $35.80     +10.2%     8.4      7

  WASH SALE BLOCKS
  ──────────────────────────────────────
  WOLF    Blocked until 2026-05-10    Loss: -$2,650
  VIAV    Blocked until 2026-04-25    Loss: -$1,800

================================================================
```

---

## Environment Variables

```bash
# Required
ALPACA_API_KEY=your-paper-api-key
ALPACA_SECRET_KEY=your-paper-secret-key

# Optional (for email alerts)
RESEND_API_KEY=your-resend-key
ALERT_EMAIL_TO=tbruschwein@gmail.com
```

Store these in a `.env` file in the project root. Use `python-dotenv` to load them:
```python
from dotenv import load_dotenv
load_dotenv()
```

**IMPORTANT:** Add `.env` to `.gitignore` so API keys are never committed to GitHub.

---

## Dependencies to Add

```
alpaca-py>=0.13
python-dotenv>=1.0
```

Add these to `requirements.txt`.

---

## Safety Measures

1. **Paper trading only** — `paper=True` is hardcoded. Add a safety check:
   ```python
   assert trading_client.get_account().account_number.startswith("PA"), "NOT A PAPER ACCOUNT — ABORTING"
   ```

2. **Max position size enforced** — never exceed 20% of equity per position

3. **Dry run mode** — add a `--dry-run` flag that scores everything and logs what it WOULD do, without placing any orders:
   ```bash
   python3 trade_executor.py --dry-run
   ```

4. **Order confirmation logging** — every order placed gets logged with Alpaca's order ID for audit trail

5. **Error handling** — if Alpaca API fails, log the error and skip that ticker (don't crash the whole run)

6. **Market hours check** — only submit orders when market is closed (after-hours). Orders will execute at next market open.

---

## Testing Steps (for Todd)

After Claude Code builds this:

1. **Set environment variables:**
   ```bash
   export ALPACA_API_KEY="your-key"
   export ALPACA_SECRET_KEY="your-secret"
   ```

2. **Run in dry-run mode first:**
   ```bash
   python3 trade_executor.py --dry-run
   ```
   This scores everything and shows what trades it would make, without placing any orders.

3. **Run for real (paper trading):**
   ```bash
   python3 trade_executor.py
   ```
   This places actual paper trades on Alpaca.

4. **Verify on Alpaca dashboard:**
   Log into https://app.alpaca.markets and check:
   - Paper Trading → Orders: see your pending/filled orders
   - Paper Trading → Positions: see your open positions
   - Paper Trading → Account: see portfolio value

5. **Run daily for 1+ month** before considering real capital.

---

## Future: Automation

Once validated, this script can be scheduled to run automatically via:
- **GitHub Actions** (free, runs on a cron schedule)
- **Railway** (where the dashboard is deployed)
- Local cron job (if computer is always on — not recommended)

The scheduling setup is a separate task — for now, run manually each evening after market close.

---

## Limitations / Known Issues

- **Crypto tickers excluded** — Alpaca handles crypto via a different API flow. Can be added later.
- **Futures not supported** — GC=F, SI=F etc. are not tradeable on Alpaca. Use ETF equivalents (GLD, SLV) which are already in our ticker list.
- **No fractional shares for all tickers** — some tickers may not support fractional shares on Alpaca. Handle gracefully by rounding down to whole shares.
- **Market orders only** — we're using market orders for simplicity. Could upgrade to limit orders later for better fill prices.
- **Alpaca paper account starts with $100k** — matches our backtest assumptions.
