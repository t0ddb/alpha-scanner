"""Cash-allocation experiment: one sizing change, unchanged execution controls."""
from dataclasses import replace
import unittest

import pandas as pd

from research.scheme_c.allocation import load_engine, sequential_sizer, verify_acceptance
from research.scheme_c.baseline import Settings, size_order


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()
        self.size = sequential_sizer(size_order)

    def test_unallocated_slots_no_longer_make_affordable_entry_too_small(self):
        # $860 investment budget spread over four empty slots is $215, below
        # the $250 gate. Sequential allocation funds the $498 target (+1 flex).
        args = (6000., 1160., 4840., 4, 10., self.cfg)
        self.assertEqual(size_order(*args), (0, "floor_or_minimum"))
        self.assertEqual(self.size(*args), (50, ""))

    def test_aggregate_floor_and_minimum_still_limit_order(self):
        self.assertEqual(self.size(6000, 600, 5400, 4, 10, self.cfg), (30, ""))
        self.assertEqual(self.size(6000, 500, 5500, 4, 10, self.cfg),
                         (0, "floor_or_minimum"))

    def test_whole_share_and_flex_unchanged(self):
        self.assertEqual(self.size(5000, 5000, 0, 12, 1000, self.cfg), (0, "whole_share"))
        self.assertEqual(self.size(5000, 5000, 0, 12, 100, self.cfg), (4, ""))
        self.assertEqual(self.size(5000, 5000, 0, 12, 10, self.cfg), (42, ""))
        self.assertEqual(self.size(5000, 5000, 0, 12, 10, replace(self.cfg, flex_pct=0)), (41, ""))

    def test_one_remaining_slot_exactly_matches_baseline(self):
        for equity in (5000, 100000):
            for cash in (100, 1000, equity):
                for price in (1, 10, 300, 10000):
                    args = (equity, cash, equity-cash, 1, price, self.cfg)
                    self.assertEqual(self.size(*args), size_order(*args))

    def test_engine_instances_do_not_change_baseline_or_each_other(self):
        control = load_engine("allocation_test_control", "baseline")
        original = control.size_order
        challenger = load_engine("allocation_test_challenger", "sequential")
        args = (6000, 1160, 4840, 4, 10, control.Settings())
        self.assertEqual(challenger.size_order(*args)[0], 50)
        self.assertEqual(control.size_order(*args)[0], 0)
        self.assertIs(control.size_order, original)
        self.assertEqual(size_order(*args)[0], 0)
        with self.assertRaisesRegex(ValueError, "allocator"):
            load_engine("bad_allocator", "typo")

    def run_fixture(self, records, **overrides):
        engine = load_engine("allocation_test_simulator", "sequential")
        days = pd.bdate_range("2024-01-02", periods=3)
        prices = pd.DataFrame({"Open": 100., "High": 101., "Low": 99.,
                               "Close": 100., "Volume": 100000.}, index=days)
        scores = pd.concat([pd.DataFrame({"date": days, "ticker": t, "score": v})
                            for t, v in records.items()])
        cfg = replace(engine.Settings(), persistence=0, **overrides)
        result = engine.simulate(scores, {t: prices for t in ["SPY"] + list(records)},
                                 "2024-01-02", "2024-01-04", 5000, cfg)
        verify_acceptance(result)
        return result

    def test_same_ranking_and_daily_entry_cap(self):
        result = self.run_fixture({t: [s]*3 for t, s in
                                   [("Z", 9), ("C", 10), ("B", 9.5), ("A", 10), ("D", 9.4)]})
        orders = [e for e in result["events"] if e["event"] == "buy_submitted"
                  and e["date"] == "2024-01-02"]
        self.assertEqual([o["ticker"] for o in orders], ["A", "C", "B", "D"])

    def test_pending_sales_still_cannot_fund_orders(self):
        result = self.run_fixture({"A": [10, 0, 0], "B": [0, 10, 10]},
                                  max_positions=1, position_pct=.9, flex_pct=0., friction_bps=0.)
        buys = [e for e in result["events"] if e["event"] == "buy_fill"]
        self.assertEqual([e["ticker"] for e in buys], ["A"])
        self.assertEqual(result["history"][1]["cash"], 500)
        self.assertGreater(result["metrics"]["counters"].get("buy_rejected_at_submission", 0), 0)

    def test_unaffordable_limit_reservation_remains_rejected_not_resized(self):
        result = self.run_fixture({"A": [10]*3}, max_positions=1, position_pct=1.,
                                  cash_floor=0., flex_pct=0., friction_bps=0.)
        self.assertFalse(any(e["event"] == "buy_fill" for e in result["events"]))
        self.assertEqual(result["metrics"]["counters"]["buy_rejected_at_submission"], 3)


if __name__ == "__main__":
    unittest.main()
