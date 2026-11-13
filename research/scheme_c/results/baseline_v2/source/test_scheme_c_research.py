"""Regression tests: run python3 -B -m unittest discover -s tests -v."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.scheme_c.baseline import (
    PriceBook, ScoreBook, Settings, path_starts, price_views, rebuild_scores,
    simulate, size_order,
)


def bars(rows, dates=None):
    dates = pd.bdate_range("2024-01-02", periods=len(rows)) if dates is None else dates
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=dates, dtype=float)
    df["Volume"] = 100_000.
    return df


def observations(dates, values, ticker="A"):
    return pd.DataFrame({"date": dates, "ticker": ticker, "score": values})


def fills(result, side):
    return [e for e in result["events"] if e["event"] == side + "_fill"]


class CausalDataTests(unittest.TestCase):
    def test_missing_price_does_not_take_future_or_past(self):
        df = bars([[100, 101, 99, 100], [300, 301, 299, 300]],
                  pd.to_datetime(["2024-01-02", "2024-01-04"]))
        pb = PriceBook({"SPY": df, "A": df})
        self.assertIsNone(pb.exact_price("A", "2024-01-03"))
        from sizing_comparison_backtest import get_price
        self.assertIsNone(get_price({"A": df}, "A", "2024-01-03"))

    def test_invalid_bar_is_not_tradeable(self):
        for value in (np.nan, np.inf, 0., -1.):
            df = bars([[100, 101, 99, 100]])
            df.loc[df.index[0], "Open"] = value
            self.assertIsNone(PriceBook({"SPY": df, "A": df}).bar("A", df.index[0]))

    def test_score_gaps_are_not_carried_forward(self):
        days = pd.bdate_range("2024-01-02", periods=8)
        df = observations([days[0]], [10])
        book = ScoreBook(df)
        self.assertNotIn(days[1], book.daily)
        self.assertFalse(book.prior_passes("A", days[7], 3, 9))
        from sizing_comparison_backtest import build_daily_scores
        self.assertEqual(build_daily_scores(df, list(days))[str(days[3].date())], {})

    def test_persistence_uses_three_distinct_prior_rows_across_gaps(self):
        days = pd.to_datetime(["2024-01-02", "2024-01-09", "2024-01-16", "2024-01-23"])
        book = ScoreBook(observations(days, [9, 9, 9, 0]))
        self.assertTrue(book.prior_passes("A", days[3], 3, 9))
        self.assertFalse(book.prior_passes("A", days[2], 3, 9))
        self.assertFalse(book.prior_passes("A", "2024-01-24", 3, 9))

    def test_duplicate_score_cannot_fabricate_persistence(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ScoreBook(observations(["2024-01-02"] * 3, [10] * 3))

    def test_invalid_scores_rejected_not_zeroed(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            ScoreBook(observations(["2024-01-02"], [np.nan]))

    def test_audit_short_history_peer_is_sliced_by_date(self):
        from audit_build_dataset import _raw_rs_by_period
        dates = pd.bdate_range("2023-01-02", periods=800)
        spy = bars([[100+i, 101+i, 99+i, 100+i] for i in range(800)], dates)
        late = spy.iloc[300:].copy()
        late.iloc[101:, :4] *= 1000
        values = _raw_rs_by_period({"LATE": late}, spy, 400, [63, 126])
        self.assertNotIn("LATE", values[126])
        self.assertAlmostEqual(values[63]["LATE"], 1.)

    def test_audit_aligns_session_labels_across_vendor_timezones(self):
        from audit_build_dataset import _raw_rs_by_period
        dates = pd.bdate_range("2023-01-02", periods=300)
        spy = bars([[100+i, 101+i, 99+i, 100+i] for i in range(300)], dates.tz_localize("America/New_York"))
        stock = spy.copy()
        stock.index = dates.tz_localize("UTC")
        values = _raw_rs_by_period({"A": stock}, spy, 299, [63])
        self.assertAlmostEqual(values[63]["A"], 1.)

    def test_audit_missing_target_is_not_nearest_future(self):
        from audit_build_dataset import build_row
        dates = pd.bdate_range("2023-01-02", periods=400)
        spy = bars([[100, 101, 99, 100]] * 400, dates)
        missing = spy.drop(dates[300])
        self.assertIsNone(build_row("A", missing, spy, 300, {}, [], {}, {}))

    def test_audit_forward_labels_use_benchmark_calendar(self):
        from audit_build_dataset import build_row
        from config import load_config
        dates = pd.bdate_range("2023-01-02", periods=400)
        all_days = pd.date_range(dates[0], dates[-1])
        close = np.arange(len(all_days), dtype=float) + 100
        crypto = bars(np.column_stack([close, close + 1, close - 1, close]), all_days)
        spy = crypto.reindex(dates)
        row = build_row("BTC-USD", crypto, spy, 300,
                        load_config("research/scheme_c/inputs/ticker_config.yaml"), [1., 2.], {}, {})
        expected = spy.Close.iloc[310] / spy.Close.iloc[300] - 1
        self.assertAlmostEqual(row["fwd_10d"], expected, places=5)

    def test_future_prices_and_peer_listing_do_not_change_past_scores(self):
        dates = pd.bdate_range("2022-01-03", periods=410)
        rng = np.random.default_rng(12)
        data = {}
        for name in ("SPY", "A", "B", "IPO"):
            close = 100 * np.exp(np.cumsum(rng.normal(.001, .02, len(dates))))
            data[name] = bars(np.column_stack([close, close * 1.02, close * .98, close]), dates)
        data["IPO"] = data["IPO"].iloc[300:]
        target = dates[330]
        a, _, _ = rebuild_scores(data, ["A", "B", "IPO"], str(dates[260].date()), str(target.date()))
        for name in data:
            data[name].loc[data[name].index > target, ["Open", "High", "Low", "Close"]] *= 10000
        b, _, _ = rebuild_scores(data, ["A", "B", "IPO"], str(dates[260].date()), str(dates[-1].date()))
        assert_frame_equal(a, b[b.date <= target].reset_index(drop=True))
        self.assertFalse((a.ticker == "IPO").any())

    def test_vectorized_scores_match_frozen_production_formulas(self):
        from indicators import compute_all_indicators, score_ticker
        from config import load_config
        from audit_build_dataset import _raw_rs_by_period
        dates = pd.bdate_range("2022-01-03", periods=340)
        rng = np.random.default_rng(32)
        data = {}
        for name in ("SPY", "A", "B", "C", "D"):
            close = 100 * np.exp(np.cumsum(rng.normal(.003, .04, len(dates))))
            data[name] = bars(np.column_stack([close, close * 1.03, close * .97, close]), dates)
        scores, _, _ = rebuild_scores(data, ["A", "B", "C", "D"], str(dates[260].date()), str(dates[-1].date()))
        cfg = load_config("research/scheme_c/inputs/ticker_config.yaml")
        for idx in (260, 274, 310, 339):
            raw = _raw_rs_by_period({t: v for t, v in data.items() if t != "SPY"}, data["SPY"], idx, [21, 63, 126])
            for ticker in ("A", "B", "C", "D"):
                multi = {f"rs_{p}d_pctl": sum(v <= raw[p][ticker] for v in raw[p].values()) / len(raw[p]) * 100
                         for p in (21, 63, 126)}
                ind = compute_all_indicators(data[ticker].iloc[:idx+1], data["SPY"].iloc[:idx+1], cfg,
                                             all_rs_values=list(raw[63].values()), multi_tf_rs=multi)
                actual = scores[(scores.ticker == ticker) & (scores.date == dates[idx])].score.iloc[0]
                self.assertEqual(actual, score_ticker(ind)["score"], (ticker, dates[idx]))


class EventClockTests(unittest.TestCase):
    def run_case(self, rows, values=None, cfg=None, capital=10_000., extra=None):
        stock = bars(rows)
        data = {"A": stock, "SPY": bars([[100, 101, 99, 100]] * len(rows))}
        if extra:
            data.update(extra)
        scores = observations(stock.index, [10.] * len(rows) if values is None else values)
        if cfg is None:
            cfg = replace(Settings(), persistence=0, friction_bps=0.)
        return simulate(scores, data, str(stock.index[0].date()), str(stock.index[-1].date()), capital, cfg)

    def test_next_open_buy_is_not_in_prior_close_equity(self):
        cfg = replace(Settings(), persistence=0, friction_bps=0, execution_constraints=False,
                      position_pct=1., max_positions=1)
        result = self.run_case([[100, 101, 99, 100], [200, 201, 199, 200]], cfg=cfg)
        self.assertEqual(result["history"][0]["equity"], 10_000)
        self.assertEqual(result["history"][0]["positions"], 0)
        self.assertEqual(result["history"][1]["equity"], 10_000)
        self.assertEqual(fills(result, "buy")[0]["date"], "2024-01-03")

    def test_score_sale_is_booked_on_fill_date(self):
        result = self.run_case([[100, 101, 99, 100], [100, 101, 99, 100], [120, 121, 119, 120]], [10, 0, 0])
        self.assertEqual(result["history"][1]["positions"], 1)
        self.assertEqual(result["history"][1]["equity"], 10_000)
        self.assertEqual(fills(result, "sell")[0]["exit_date"], "2024-01-04")
        self.assertEqual(fills(result, "sell")[0]["exit_price"], 120)

    def test_existing_intraday_stop_fills_that_session_at_stop(self):
        result = self.run_case([[100, 101, 99, 100], [100, 101, 99, 100],
                               [100, 101, 70, 100], [100, 101, 99, 100]])
        trade = fills(result, "sell")[0]
        self.assertEqual((trade["exit_date"], trade["exit_price"], trade["reason"]),
                         ("2024-01-04", 80, "stop_intraday"))

    def test_gap_through_stop_executes_at_open_below_stop(self):
        result = self.run_case([[100, 101, 99, 100], [100, 101, 99, 100], [60, 70, 55, 65]])
        self.assertEqual(fills(result, "sell")[0]["exit_price"], 60)
        self.assertEqual(fills(result, "sell")[0]["reason"], "stop_gap")

    def test_intraday_stop_precedes_same_day_low_score(self):
        result = self.run_case([[100, 101, 99, 100], [100, 101, 99, 100], [100, 101, 70, 90]], [10, 10, 0])
        self.assertEqual(fills(result, "sell")[0]["reason"], "stop_intraday")

    def test_fill_session_low_cannot_trigger_unattached_stop(self):
        result = self.run_case([[100, 101, 99, 100], [100, 101, 60, 100], [100, 101, 99, 100]])
        self.assertEqual(fills(result, "sell"), [])
        attached = [e for e in result["events"] if e["event"] == "stop_attached_after_close"]
        self.assertEqual(attached[0]["date"], "2024-01-03")

    def test_fill_session_close_below_stop_queues_market_exit(self):
        result = self.run_case([[100, 101, 99, 100], [100, 101, 60, 70], [60, 65, 55, 60]])
        self.assertEqual(fills(result, "sell")[0]["reason"], "close_stop_exit")
        self.assertEqual(fills(result, "sell")[0]["exit_price"], 60)

    def test_day_limit_can_fill_after_above_limit_open(self):
        result = self.run_case([[100, 101, 99, 100], [110, 112, 102, 109]])
        self.assertEqual(fills(result, "buy")[0]["price"], 103)
        self.assertEqual(fills(result, "buy")[0]["phase"], "intraday_touch")

    def test_limit_open_price_improvement(self):
        result = self.run_case([[100, 101, 99, 100], [95, 101, 94, 100]])
        self.assertEqual(fills(result, "buy")[0]["price"], 95)

    def test_unfilled_day_limit_expires_and_never_creates_holding(self):
        result = self.run_case([[100, 101, 99, 100], [110, 112, 105, 109]])
        self.assertEqual(fills(result, "buy"), [])
        self.assertEqual(result["history"][1]["cash"], 10_000)
        self.assertEqual(result["metrics"]["counters"]["buy_expired"], 1)

    def test_missing_score_does_not_force_exit(self):
        stock = bars([[100, 101, 99, 100]] * 4)
        result = simulate(observations([stock.index[0]], [10]), {"SPY": stock, "A": stock},
                          str(stock.index[0].date()), str(stock.index[-1].date()),
                          cfg=replace(Settings(), persistence=0))
        self.assertEqual(fills(result, "sell"), [])
        self.assertEqual(len(fills(result, "buy")), 1)

    def test_persistence_can_use_real_history_before_path_start(self):
        stock = bars([[100, 101, 99, 100]] * 6)
        result = simulate(observations(stock.index, [10]*6), {"SPY": stock, "A": stock},
                          str(stock.index[3].date()), str(stock.index[-1].date()))
        self.assertEqual(fills(result, "buy")[0]["signal_date"], str(stock.index[3].date()))
        self.assertEqual(fills(result, "buy")[0]["date"], str(stock.index[4].date()))

    def test_missing_held_price_fails_loudly_no_stale_valuation(self):
        stock = bars([[100, 101, 99, 100]] * 4)
        with self.assertRaisesRegex(ValueError, "Missing/invalid held bar"):
            simulate(observations(stock.index, [10]*4), {"SPY": stock, "A": stock.drop(stock.index[2])},
                     str(stock.index[0].date()), str(stock.index[-1].date()),
                     cfg=replace(Settings(), persistence=0))

    def test_no_end_of_window_phantom_liquidation_or_fill(self):
        result = self.run_case([[100, 101, 99, 100]])
        self.assertEqual(fills(result, "buy"), [])
        self.assertEqual(result["metrics"]["pending_orders"], 1)
        result = self.run_case([[100, 101, 99, 100]] * 2)
        self.assertEqual(result["metrics"]["open_positions"], 1)
        self.assertEqual(result["closed_trades"], [])

    def test_append_future_prices_does_not_change_past_events_or_marks(self):
        short = self.run_case([[100, 101, 99, 100]] * 3)
        long = self.run_case([[100, 101, 99, 100]] * 3 + [[200, 300, 1, 250]])
        self.assertEqual(short["history"], long["history"][:3])
        self.assertEqual(short["events"], [e for e in long["events"] if e["date"] <= "2024-01-04"])


class ExecutionConstraintsTests(unittest.TestCase):
    def test_whole_share_flex_preserves_existing_formula(self):
        qty, reason = size_order(5000, 5000, 0, 12, 235, Settings())
        self.assertEqual((qty, reason), (2, ""))
        # Whole-share/minimum gate order is intentional: no rounding from zero.
        self.assertEqual(size_order(5000, 5000, 0, 12, 500, Settings())[0], 0)

    def test_minimum_gate_precedes_share_rounding(self):
        self.assertEqual(size_order(5000, 5000, 0, 12, 250, replace(Settings(), min_position=500))[0], 0)
        qty, _ = size_order(5000, 5000, 0, 12, 210, replace(Settings(), flex_pct=0))
        self.assertEqual(qty, 1)  # filled $210 can be below the $250 pre-round gate.

    def test_daily_cap_and_exclusions(self):
        df = bars([[100, 101, 99, 100]] * 2)
        names = ["A", "B", "C", "D", "E", "BTC-USD", "GC=F", "SI=F", "BLOCKED"]
        scores = pd.concat([observations(df.index, [10, 10], t) for t in names])
        result = simulate(scores, {t: df for t in names + ["SPY"]}, "2024-01-02", "2024-01-03",
                          cfg=replace(Settings(), persistence=0), blocked=["BLOCKED"])
        buys = fills(result, "buy")
        self.assertEqual([e["ticker"] for e in buys], ["A", "B", "C", "D"])
        self.assertTrue(all(e["qty"] == int(e["qty"]) for e in buys))

    def test_cap_and_cash_invariants_across_multiple_order_batches(self):
        df = bars([[100, 101, 99, 100]] * 8)
        names = [f"T{i:02d}" for i in range(25)]
        scores = pd.concat([observations(df.index, [10]*len(df), t) for t in names])
        result = simulate(scores, {t: df for t in names + ["SPY"]},
                          str(df.index[0].date()), str(df.index[-1].date()),
                          cfg=replace(Settings(), persistence=0))
        self.assertTrue(all(h["positions"] <= 12 and h["cash"] >= 0 for h in result["history"]))
        self.assertTrue(all(sum(e["date"] == d for e in fills(result, "buy")) <= 4
                            for d in [str(day.date()) for day in df.index]))
        for h in result["history"]:
            self.assertAlmostEqual(h["equity"], h["cash"] + h["invested"] + h["dividend_receivable"])

    def test_limit_notional_reserved_not_just_sizing_estimate(self):
        df = bars([[100, 101, 99, 100]] * 2)
        result = simulate(observations(df.index, [10]*2), {"SPY": df, "A": df},
                          str(df.index[0].date()), str(df.index[-1].date()),
                          cfg=replace(Settings(), persistence=0, position_pct=1., max_positions=1,
                                      cash_floor=0., flex_pct=0., friction_bps=0.))
        self.assertEqual(fills(result, "buy"), [])
        self.assertEqual(result["metrics"]["counters"]["buy_rejected_at_submission"], 2)

    def test_daily_loss_halt_gates_entries_only(self):
        stock = bars([[100, 101, 99, 100], [100, 101, 40, 50], [40, 41, 39, 40]])
        steady = bars([[100, 101, 99, 100]] * 3)
        scores = pd.concat([observations(stock.index, [10, 0, 0], "A"),
                            observations(stock.index, [0, 10, 10], "B")])
        cfg = replace(Settings(), persistence=0, max_positions=1, position_pct=.9,
                      cash_floor=.05, flex_pct=0, friction_bps=0)
        result = simulate(scores, {"SPY": steady, "A": stock, "B": steady}, "2024-01-02", "2024-01-04", cfg=cfg)
        self.assertGreater(result["metrics"]["counters"].get("skip_daily_loss_halt", 0), 0)
        self.assertEqual(fills(result, "sell")[0]["ticker"], "A")
        self.assertFalse(any(e["ticker"] == "B" for e in fills(result, "buy")))

    def test_sale_proceeds_not_available_at_previous_close(self):
        df = bars([[100, 101, 99, 100]] * 3)
        scores = pd.concat([observations(df.index, [10, 0, 0], "A"),
                            observations(df.index, [0, 10, 10], "B")])
        cfg = replace(Settings(), persistence=0, max_positions=1, position_pct=.9,
                      flex_pct=0, friction_bps=0)
        result = simulate(scores, {"SPY": df, "A": df, "B": df}, "2024-01-02", "2024-01-04", cfg=cfg)
        self.assertEqual(result["history"][1]["cash"], 10_000)
        self.assertEqual(len(fills(result, "buy")), 1)
        self.assertGreater(result["metrics"]["counters"].get("buy_rejected_at_submission", 0), 0)

    def test_friction_decreases_account_value(self):
        df = bars([[100, 101, 99, 100]] * 3)
        args = (observations(df.index, [10, 0, 0]), {"SPY": df, "A": df}, "2024-01-02", "2024-01-04")
        zero = simulate(*args, cfg=replace(Settings(), persistence=0, friction_bps=0))
        cost = simulate(*args, cfg=replace(Settings(), persistence=0, friction_bps=25))
        self.assertLess(cost["metrics"]["final_equity"], zero["metrics"]["final_equity"])
        self.assertAlmostEqual(cost["metrics"]["final_equity"], 100_000 - cost["metrics"]["fees"])

    def test_split_does_not_create_loss_and_dividend_not_spendable(self):
        df = bars([[50, 51, 49, 50]] * 4)
        df["Adj Close"] = df.Close
        df["Stock Splits"] = [0, 0, 2, 0]
        df["Dividends"] = [0, 0, 0, 1]
        adjusted, raw = price_views({"SPY": df, "A": df})
        self.assertEqual(raw["A"].Open.iloc[0], 100)
        self.assertEqual(raw["A"].Open.iloc[2], 50)
        result = simulate(observations(df.index, [10]*4), raw, "2024-01-02", "2024-01-05",
                          cfg=replace(Settings(), persistence=0, friction_bps=0))
        hist = result["history"]
        self.assertEqual(hist[1]["equity"], hist[2]["equity"])
        self.assertEqual(hist[2]["cash"], hist[3]["cash"])
        qty = hist[3]["holdings"]["A"]["qty"]
        self.assertEqual(hist[3]["dividend_receivable"], qty)
        self.assertEqual(fills(result, "sell"), [])

    def test_path_starts_respect_explicit_window_and_count(self):
        calendar = pd.bdate_range("2023-01-02", "2026-12-31")
        starts = path_starts(calendar, "2024-01-02", "2024-06-30", 20)
        self.assertEqual(len(set(starts)), 20)
        self.assertEqual(starts[0], "2024-01-02")
        self.assertLessEqual(starts[-1], "2024-06-30")
        with self.assertRaisesRegex(ValueError, "Requested"):
            path_starts(calendar, "2024-01-02", "2024-01-03", 20)


if __name__ == "__main__":
    unittest.main()
