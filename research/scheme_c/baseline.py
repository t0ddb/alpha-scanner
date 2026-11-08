"""Offline Scheme C research primitives. No brokerage or production state imports.

See docs/SCHEME_C_RESEARCH_BASELINE.md for the event clock and limitations.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

OHLC = ["Open", "High", "Low", "Close"]


def normalize(frame):
    df = frame.copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    return df


def price_views(vendor):
    """Return adjusted signal bars and historical share-denomination bars.

    Yahoo's unadjusted OHLC is still split-adjusted. Undo subsequent splits
    for execution only; apply splits to outstanding shares/orders on their
    effective dates. No future prices or future actions enter signal formulas.
    """
    signals, execution = {}, {}
    for ticker, frame in sorted(vendor.items()):
        df = normalize(frame)
        adjusted = df[OHLC + ["Volume"]].copy()
        ratio = df["Adj Close"] / df["Close"]
        adjusted[OHLC] = adjusted[OHLC].mul(ratio, axis=0)
        splits = df.get("Stock Splits", pd.Series(0., index=df.index)).replace(0, 1.)
        future_factor = splits.iloc[::-1].cumprod().iloc[::-1] / splits
        raw = df[OHLC + ["Volume"]].copy()
        raw[OHLC] = raw[OHLC].mul(future_factor, axis=0)
        raw["Volume"] = raw["Volume"] / future_factor
        raw["split"] = splits
        raw["dividend"] = df.get("Dividends", 0.) * future_factor
        signals[ticker], execution[ticker] = adjusted, raw
    return signals, execution


def valid_bars(df):
    values = df[OHLC].to_numpy(dtype=float)
    ok = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    tol = df["High"].abs() * 1e-7
    ok &= (df["High"] + tol >= df[["Open", "Close", "Low"]].max(axis=1)).to_numpy()
    ok &= (df["Low"] - tol <= df[["Open", "Close", "High"]].min(axis=1)).to_numpy()
    return pd.Series(ok, index=df.index)


def rebuild_scores(prices, universe, start, end, min_history=261):
    """Vectorized, causal implementation of the frozen production indicators.

    All rolling windows use SPY sessions, including cross-sectional peers.
    Missing bars remain missing. A peer needs a complete horizon; scoring also
    requires 261 observed bars (the prior audit's warm-up) and 127 complete
    recent sessions (the longest active indicator). No forward labels exist.
    """
    calendar = normalize(prices["SPY"]).index
    calendar = calendar[calendar <= pd.Timestamp(end)]
    frames = {}
    for ticker in ["SPY"] + sorted(set(universe)):
        df = normalize(prices[ticker]).reindex(calendar)
        df.loc[~valid_bars(df), OHLC] = np.nan
        frames[ticker] = df
    names = sorted(set(universe))
    closes = pd.DataFrame({t: frames[t]["Close"] for t in names}, index=calendar)
    spy = frames["SPY"]["Close"]
    ranks, peer_counts = {}, {}
    for period in (21, 63, 126):
        sr = closes / closes.shift(period) - 1
        br = spy / spy.shift(period) - 1
        complete = closes.rolling(period + 1).count().eq(period + 1)
        bench_ok = spy.rolling(period + 1).count().eq(period + 1)
        # Preserve the existing RS formula, including its zero-benchmark case.
        raw = sr.div(br.where(br > 0), axis=0)
        raw.loc[br < 0] = sr.sub(br, axis=0).loc[br < 0]
        raw.loc[br == 0] = 0.
        raw = raw.where(complete).where(bench_ok, axis=0)
        ranks[period] = raw.rank(axis=1, method="max", pct=True) * 100
        peer_counts[period] = raw.count(axis=1)
    rows, coverage = [], []
    for ticker in names:
        df = frames[ticker]
        high, low, close = df.High, df.Low, df.Close
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        ich = (close > pd.concat([span_a, span_b], axis=1).max(axis=1)) & (span_a > span_b)
        period_low = low.rolling(5).min()
        higher_lows = pd.Series(0, index=calendar)
        # Same oldest-to-newest recurrence as check_higher_lows, max count 4.
        for back in (3, 2, 1, 0):
            higher_lows = (higher_lows + 1).where(
                period_low.shift(back * 5) > period_low.shift((back + 1) * 5), 0)
        roc = close / close.shift(21) * 100 - 100
        tr = pd.concat([high - low, (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pctl = atr.rolling(50).apply(lambda v: np.mean(v <= v[-1]) * 100, raw=True)
        p21, p63, p126 = (ranks[p][ticker] for p in (21, 63, 126))
        # Production rounds primary RS percentile before its gradient lookup;
        # the Dual-TF test uses unrounded ranks.
        rs_pctl = p63.round(1).where(peer_counts[63] > 1, 0)
        rs_points = np.select([rs_pctl >= n for n in (90, 80, 70, 60, 50)],
                              [3., 2.4, 1.8, 1.2, .6], default=0.)
        hl_points = np.select([higher_lows >= n for n in (5, 4, 3, 2)],
                              [.5, .375, .25, .125], default=0.)
        dtf = ((p126 >= 70) & (p63 > p126)) | ((p63 >= 80) & (p21 >= 80))
        points = (rs_points + hl_points + ich * 2. + (roc > 5) * 1.5
                  + dtf * 2.5 + (atr_pctl >= 80) * .5)
        observed = valid_bars(df) & np.isfinite(df.Volume) & (df.Volume >= 0)
        eligible = (observed.cumsum() >= min_history) & observed.rolling(127).sum().eq(127)
        eligible &= spy.rolling(127).count().eq(127)
        out = pd.DataFrame({"date": calendar, "ticker": ticker,
                            "score": [round(float(v), 1) for v in points],
                            "score_legacy_rounding": [round(float(v), 2) for v in points],
                            "rs_percentile": rs_pctl.to_numpy(),
                            "rs_21d_pctl": p21.to_numpy(), "rs_63d_pctl": p63.to_numpy(),
                            "rs_126d_pctl": p126.to_numpy(),
                            "higher_lows_count": higher_lows.to_numpy(),
                            "ichimoku_fired": ich.to_numpy(), "roc_fired": (roc > 5).to_numpy(),
                            "dual_tf_rs_fired": dtf.to_numpy(),
                            "atr_fired": (atr_pctl >= 80).to_numpy(),
                            "bars_observed": observed.cumsum().to_numpy()})
        select = eligible.to_numpy() & (calendar >= pd.Timestamp(start))
        rows.append(out.loc[select])
        coverage.append({"ticker": ticker, "score_rows": int(select.sum()),
                         "first_score": str(calendar[select][0].date()) if select.any() else None,
                         "last_score": str(calendar[select][-1].date()) if select.any() else None,
                         "invalid_or_absent_sessions": int((~observed).sum())})
    scores = pd.concat(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    peers = pd.DataFrame({"date": calendar, **{f"peers_{p}": peer_counts[p].to_numpy()
                                              for p in (21, 63, 126)}})
    return scores, pd.DataFrame(coverage), peers


class ScoreBook:
    """Exact observations only; missingness is neither a score nor an exit."""
    def __init__(self, scores):
        df = scores[["date", "ticker", "score"]].copy()
        df["date"] = pd.to_datetime(df.date).dt.normalize()
        if df.duplicated(["date", "ticker"]).any():
            raise ValueError("Duplicate score observations")
        if not np.isfinite(df.score).all():
            raise ValueError("Non-finite scores must be omitted, not imputed")
        df = df.sort_values(["date", "ticker"])
        self.daily = {d: dict(zip(g.ticker, g.score)) for d, g in df.groupby("date")}
        self.history = {t: (pd.DatetimeIndex(g.date), g.score.to_numpy())
                        for t, g in df.groupby("ticker")}

    def prior_passes(self, ticker, day, count, threshold):
        if count == 0:
            return True
        if ticker not in self.history:
            return False
        dates, values = self.history[ticker]
        i = dates.searchsorted(pd.Timestamp(day), side="left")
        return i >= count and bool((values[i-count:i] >= threshold).all())


class PriceBook:
    def __init__(self, prices):
        self.frames = {t: normalize(d) for t, d in prices.items()}
        self.calendar = self.frames["SPY"].index
        self.bars = {}
        for ticker, df in self.frames.items():
            for d, row in df.loc[valid_bars(df)].iterrows():
                self.bars[ticker, d] = {k: float(row[k]) for k in OHLC}
                self.bars[ticker, d].update(
                    split=float(row.get("split", 1.)), dividend=float(row.get("dividend", 0.)),
                    Volume=float(row.get("Volume", 0.)))

    def bar(self, ticker, day):
        return self.bars.get((ticker, pd.Timestamp(day)))

    def exact_price(self, ticker, day, column="Open"):
        bar = self.bar(ticker, day)
        return None if bar is None else bar[column]


@dataclass(frozen=True)
class Settings:
    entry_threshold: float = 9.
    exit_threshold: float = 5.
    persistence: int = 3
    stop_pct: float = .20
    max_positions: int = 12
    position_pct: float = .083
    flex_pct: float = .012
    cash_floor: float = .05
    min_position: float = 250.
    max_entries_per_day: int = 4
    limit_buffer: float = .03
    daily_loss_halt: float = .10
    # Modeling assumptions, not optimized trading parameters:
    friction_bps: float = 5.
    # 'touch' fills a DAY limit if Low <= limit; 'open_only' is a sensitivity.
    limit_model: str = "touch"
    # Constraints off is a diagnostic bridge to the original simulator only.
    execution_constraints: bool = True
    # Conservative broker-acceptance proxy: reserve limit notional from cash
    # actually available at submission; projected exit proceeds cannot fund it.
    funding: str = "cash_at_submission"

    def __post_init__(self):
        if self.limit_model not in {"touch", "open_only"}:
            raise ValueError("Unknown limit model")
        if self.funding not in {"cash_at_submission", "next_open_sales"}:
            raise ValueError("Unknown funding model")


def excluded(ticker):
    return ticker.endswith("-USD") or ticker in {"GC=F", "SI=F"}


def size_order(equity, projected_cash, committed, slots, price, cfg):
    """Production's estimate-time floor, min gate, integer floor, +1 flex.

    The min gate precedes rounding; it is NOT a minimum filled notional.
    Flex can exceed the per-slot floor cap, as in the existing implementation.
    """
    raw = equity * cfg.position_pct
    budget = equity * (1 - cfg.cash_floor) - committed
    cap = max(0., budget / max(1, slots))
    target = min(projected_cash, raw, cap)
    if target < cfg.min_position:
        return 0, "floor_or_minimum"
    qty = int(target // price)
    if qty == 0:
        return 0, "whole_share"
    cost = (qty + 1) * price
    if (cfg.flex_pct > 0 and cost <= equity * (cfg.position_pct + cfg.flex_pct)
            and cost <= projected_cash and cost + committed <= equity * (1 - cfg.cash_floor)):
        qty += 1
    return qty, ""


@dataclass
class Position:
    ticker: str
    qty: float
    entry_date: str
    entry_price: float
    cost_basis: float
    entry_fee: float
    stop: float
    mark: float
    mark_date: str
    stop_active: bool = False
    dividend_income: float = 0.


def simulate(scores, prices, start, end, capital=100_000., cfg=Settings(), blocked=()):
    """EOD signals -> queued orders -> next-session fills -> close marks.

    The research batch is assumed to run at 23:59 New York time, after all
    same-date equity/crypto/futures daily bars are final and before the next
    equity session. Regular-session Close is the explicit sizing/mark proxy.
    Daily bars cannot reconstruct a 16:35 run's unfinished non-equity peers.

    Active stops fill before close signals. Entry stops are attached after the
    fill session, matching the existing daily pipeline. No terminal liquidation
    is invented. Missing held bars fail loudly: an unknown path through a stop
    cannot silently be treated as a day on which the stop survived.
    """
    book = scores if isinstance(scores, ScoreBook) else ScoreBook(scores)
    pb = prices if isinstance(prices, PriceBook) else PriceBook(prices)
    days = pb.calendar[(pb.calendar >= pd.Timestamp(start)) & (pb.calendar <= pd.Timestamp(end))]
    if not len(days) or days[0] != pd.Timestamp(start) or days[-1] != pd.Timestamp(end):
        raise ValueError("Explicit start/end must both be benchmark sessions")
    cash, receivable = float(capital), 0.
    positions, pending = {}, []
    ledger, snapshots, closed = [], [], []
    counters = Counter()
    previous_equity = float(capital)
    fee_rate = cfg.friction_bps / 10_000
    order_id = 0

    def event(day, kind, **kw):
        ledger.append({"date": str(day.date()), "event": kind, **kw})
        counters[kind] += 1

    def sell(day, ticker, price, reason, oid=None):
        nonlocal cash
        pos = positions.pop(ticker)
        proceeds = pos.qty * price
        fee = proceeds * fee_rate
        cash += proceeds - fee
        pnl = proceeds - fee - pos.cost_basis - pos.entry_fee + pos.dividend_income
        trade = {"ticker": ticker, "entry_date": pos.entry_date,
                 "exit_date": str(day.date()), "entry_price": pos.entry_price,
                 "exit_price": price, "qty": pos.qty, "reason": reason,
                 "pnl": pnl, "return_pct": pnl / (pos.cost_basis + pos.entry_fee) * 100,
                 "hold_days": (day - pd.Timestamp(pos.entry_date)).days}
        closed.append(trade)
        event(day, "sell_fill", order_id=oid, **trade, fee=fee, cash_after=cash)

    for day in days:
        ds = str(day.date())
        # 1. Corporate actions, before the open; dividends go to receivables,
        # not spendable cash (payment dates are absent from daily vendor bars).
        for ticker, pos in list(positions.items()):
            bar = pb.bar(ticker, day)
            if bar is None:
                raise ValueError(f"Missing/invalid held bar: {ticker} {ds}; cannot resolve stop or valuation")
            ratio = bar["split"]
            if ratio != 1:
                pos.qty *= ratio
                pos.entry_price /= ratio
                pos.stop /= ratio
                pos.mark /= ratio
                event(day, "split", ticker=ticker, ratio=ratio, qty=pos.qty)
            if bar["dividend"]:
                amount = pos.qty * bar["dividend"]
                receivable += amount
                pos.dividend_income += amount
                event(day, "dividend_accrued", ticker=ticker, amount=amount)
        # Adjust pending split-date orders. Fractional entitlements are kept
        # after splits; all NEW entries are integer shares in constrained runs.
        for order in pending:
            bar = pb.bar(order["ticker"], day)
            if bar and bar["split"] != 1 and order["side"] == "buy":
                order["qty"] = math.floor(order["qty"] * bar["split"])
                order["limit"] = round(order["limit"] / bar["split"], 2)
                event(day, "order_split_adjustment", order_id=order["id"], ticker=order["ticker"])
        # 2. Pending score/close-stop exits fill at this open. Their old stops
        # were canceled at submission, so they cannot also fire today.
        for order in [o for o in pending if o["side"] == "sell"]:
            ticker = order["ticker"]
            bar = pb.bar(ticker, day)
            if bar is None:
                raise ValueError(f"Missing market-exit bar: {ticker} {ds}")
            sell(day, ticker, bar["Open"], order["reason"], order["id"])
        # 3. Existing stops gapping through at the open.
        for ticker, pos in list(positions.items()):
            bar = pb.bar(ticker, day)
            if pos.stop_active and bar["Open"] <= pos.stop:
                sell(day, ticker, bar["Open"], "stop_gap")
        # 4. Opening limit fills, then existing intraday stops, then intraday
        # limit touches. This is a stated daily-bar convention. Orders reserve
        # cash at submission, so unknown intraday sale ordering cannot fund buys.
        buys = [o for o in pending if o["side"] == "buy"]

        def buy(order, price, phase):
            nonlocal cash
            ticker, qty = order["ticker"], order["qty"]
            if not cfg.execution_constraints:
                qty = order["budget"] / price
            cost = qty * price
            fee = cost * fee_rate
            if qty <= 0 or cost + fee > cash + 1e-8 or len(positions) >= cfg.max_positions:
                event(day, "buy_rejected_at_fill", order_id=order["id"], ticker=ticker,
                      reason="cash_or_capacity", cash=cash, required=cost + fee)
                return
            cash -= cost + fee
            stop = round(price * (1 - cfg.stop_pct), 2) if cfg.execution_constraints else price * (1 - cfg.stop_pct)
            positions[ticker] = Position(ticker, qty, ds, price, cost, fee, stop, price, ds)
            event(day, "buy_fill", order_id=order["id"], signal_date=order["signal_date"],
                  ticker=ticker, qty=qty, price=price, fee=fee, phase=phase, cash_after=cash)

        intraday = []
        for order in buys:
            bar = pb.bar(order["ticker"], day)
            if bar is None or bar["Volume"] <= 0:
                event(day, "buy_expired", order_id=order["id"], ticker=order["ticker"], reason="missing_or_zero_volume")
            elif not cfg.execution_constraints or bar["Open"] <= order["limit"]:
                buy(order, bar["Open"], "open")
            elif cfg.limit_model == "touch" and bar["Low"] <= order["limit"]:
                intraday.append(order)
            else:
                event(day, "buy_expired", order_id=order["id"], ticker=order["ticker"], reason="limit_not_reached")
        for ticker, pos in list(positions.items()):
            bar = pb.bar(ticker, day)
            if pos.stop_active and bar["Low"] <= pos.stop:
                sell(day, ticker, pos.stop, "stop_intraday")
        for order in intraday:
            buy(order, order["limit"], "intraday_touch")
        pending = []
        # 5. Closing valuation contains only executed positions, actual cash,
        # and earned dividend receivables. Never tomorrow's transaction prices.
        for ticker, pos in positions.items():
            pos.mark = pb.bar(ticker, day)["Close"]
            pos.mark_date = ds
        invested = sum(p.qty * p.mark for p in positions.values())
        equity = cash + invested + receivable
        snapshot = {"date": ds, "equity": equity, "cash": cash, "invested": invested,
                    "dividend_receivable": receivable, "positions": len(positions),
                    "cash_fraction": cash / equity,
                    "holdings": {t: {"qty": p.qty, "mark": p.mark, "mark_date": p.mark_date}
                                 for t, p in sorted(positions.items())}}
        snapshots.append(snapshot)
        if cash < -1e-7 or len(positions) > cfg.max_positions:
            raise AssertionError("Cash/cap invariant violated")
        today = book.daily.get(day, {})
        # 6. Attach stops at the daily run AFTER the entry session. A newly
        # filled position below its stop at close queues the existing fallback
        # market-exit rule; earlier fill-session lows cannot trigger retroactively.
        exits = []
        for ticker, pos in positions.items():
            score = today.get(ticker)
            reason = "score_exit" if score is not None and score < cfg.exit_threshold else None
            if reason is None and pos.mark <= pos.stop:
                reason = "close_stop_exit"
            if reason is not None:
                exits.append(ticker)
                pos.stop_active = False
                order_id += 1
                order = {"id": order_id, "side": "sell", "ticker": ticker,
                         "signal_date": ds, "reason": reason}
                pending.append(order)
                event(day, "sell_submitted", **order)
            elif not pos.stop_active:
                pos.stop_active = True
                event(day, "stop_attached_after_close", ticker=ticker, stop=pos.stop)
        # 7. Production's projected sizing math is an estimate, NOT booked
        # cash. Acceptance below uses real cash/limit commitments separately.
        projected_cash = cash + sum(positions[t].qty * positions[t].mark for t in exits)
        acceptance_cash = cash if cfg.funding == "cash_at_submission" else projected_cash
        committed = max(0., equity - projected_cash)
        count = len(positions) - len(exits)
        accepted, reserved = 0, 0.
        halted = cfg.execution_constraints and (previous_equity - equity) / previous_equity > cfg.daily_loss_halt
        candidates = sorted(today.items(), key=lambda pair: (-pair[1], pair[0]))
        for ticker, score in candidates:
            if score < cfg.entry_threshold or ticker in positions:
                continue
            if not book.prior_passes(ticker, day, cfg.persistence, cfg.entry_threshold):
                counters["skip_persistence"] += 1
                continue
            if cfg.execution_constraints and (excluded(ticker) or ticker in blocked):
                counters["skip_tradeability"] += 1
                continue
            if halted:
                counters["skip_daily_loss_halt"] += 1
                continue
            if count >= cfg.max_positions:
                counters["skip_capacity"] += 1
                continue
            if cfg.execution_constraints and cfg.max_entries_per_day and accepted >= cfg.max_entries_per_day:
                counters["skip_daily_cap"] += 1
                continue
            price = pb.exact_price(ticker, day, "Close")
            if price is None:
                counters["skip_missing_signal_price"] += 1
                continue
            limit = round(price * (1 + cfg.limit_buffer), 2)
            if cfg.execution_constraints:
                qty, reason = size_order(equity, projected_cash, committed,
                                         cfg.max_positions - count, price, cfg)
                if not qty:
                    counters["skip_" + reason] += 1
                    continue
                reserve = qty * limit * (1 + fee_rate)
                if reserve > acceptance_cash - reserved + 1e-8:
                    event(day, "buy_rejected_at_submission", ticker=ticker, reason="limit_cash_reservation",
                          required=reserve, available=acceptance_cash-reserved)
                    continue
                budget = qty * price
            else:
                budget = min(equity * cfg.position_pct, projected_cash / (1 + fee_rate))
                if budget <= 0:
                    continue
                qty, reserve = budget / price, budget * (1 + fee_rate)
            order_id += 1
            order = {"id": order_id, "side": "buy", "ticker": ticker, "qty": qty,
                     "limit": limit, "budget": budget, "signal_date": ds, "score": score}
            pending.append(order)
            event(day, "buy_submitted", **order, estimated_price=price, reserved=reserve)
            projected_cash -= budget
            committed += budget
            reserved += reserve
            count += 1
            accepted += 1
        previous_equity = equity
    # End-of-window orders remain unexecuted; holdings remain marked, not sold.
    values = np.array([capital] + [h["equity"] for h in snapshots], dtype=float)
    returns = values[1:] / values[:-1] - 1
    # Match the review's 4% annual risk-free convention for descriptive Sharpe.
    excess = returns - .04 / 252
    std = excess.std(ddof=0)
    metrics = {
        "start": start, "end": end, "capital": capital,
        "total_return": (values[-1] / capital - 1) * 100,
        "sharpe": float(excess.mean() / std * np.sqrt(252)) if std > 1e-12 else None,
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1) * 100),
        "final_equity": values[-1], "closed_trades": len(closed),
        "win_rate": float(np.mean([t["pnl"] > 0 for t in closed]) * 100) if closed else None,
        "avg_hold": float(np.mean([t["hold_days"] for t in closed])) if closed else None,
        "open_positions": len(positions), "pending_orders": len(pending),
        "minimum_cash": min(h["cash"] for h in snapshots),
        "cash_floor_breach_days": sum(h["cash_fraction"] < cfg.cash_floor - 1e-9 for h in snapshots),
        "dividend_receivable": receivable, "fees": sum(e.get("fee", 0.) for e in ledger),
        "counters": dict(sorted(counters.items())),
    }
    return {"settings": asdict(cfg), "metrics": metrics, "history": snapshots,
            "events": ledger, "closed_trades": closed, "pending_orders": pending}


def path_starts(calendar, first, last, count=20, stride=5):
    """Explicit, bounded distinct sessions; never infer a window from all scores."""
    eligible = calendar[(calendar >= pd.Timestamp(first)) & (calendar <= pd.Timestamp(last))]
    starts = eligible[::stride][:count]
    if len(starts) != count:
        raise ValueError(f"Requested {count} starts but only {len(starts)} fit the explicit window")
    return [str(d.date()) for d in starts]
