from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.signal_generator import generate_signal


class SignalGeneratorTests(unittest.TestCase):
    def test_generate_signal_rejects_stale_factor_cache(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01"},
            "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ):
            with self.assertRaises(ValueError):
                generate_signal("2024-01-03", previous_holdings=[])

    def test_generate_signal_latest_rejects_stale_data_before_required_date(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ), patch("src.signal_generator.required_latest_data_date", return_value=pd.Timestamp("2024-01-03")):
            with self.assertRaisesRegex(
                ValueError,
                "latest factor date 2024-01-02.*latest score date 2024-01-02.*required data date 2024-01-03",
            ):
                generate_signal("latest", previous_holdings=[])

    def test_generate_signal_latest_accepts_data_at_required_date(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-02"},
            "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ), patch("src.signal_generator.required_latest_data_date", return_value=pd.Timestamp("2024-01-02")):
            signal, holdings = generate_signal("latest", previous_holdings=[])

        self.assertEqual(holdings, ["E"])
        self.assertEqual(signal["date"].unique().tolist(), ["2024-01-02"])

    def test_generate_signal_specific_date_still_requires_matching_latest_cache_date(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-02")], ["A", "B", "C", "D", "E"]],
            names=["datetime", "instrument"],
        )
        factors = pd.DataFrame({"ROC5": range(1, 6)}, index=index)
        config = {
            "data": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            "strategy": {"factor_group": "momentum", "top_n": 1, "max_turnover": 1, "rank_buffer": 0},
            "factors": {"cache_file": "unused.parquet"},
            "outputs": {"holdings_file": "unused.csv"},
        }

        with patch("src.signal_generator.load_config", return_value=config), patch(
            "src.signal_generator.load_or_compute_factors", return_value=factors
        ):
            signal, holdings = generate_signal("2024-01-02", previous_holdings=[])

        self.assertEqual(holdings, ["E"])
        self.assertEqual(signal["date"].unique().tolist(), ["2024-01-02"])


if __name__ == "__main__":
    unittest.main()
