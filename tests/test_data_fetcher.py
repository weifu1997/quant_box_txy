from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.data_fetcher import (
    DAILY_FIELDS,
    fetch_daily_stocks,
    fetch_stock_universe,
    fetch_trade_calendar,
    filter_universe_frame,
    required_latest_data_date,
    update_daily_data,
    update_daily_data_resumable,
)


class FakeTushareClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, list[str] | str | None]] = []

    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        codes = str(params.get("ts_code", "")).split(",")
        rows = []
        for code in codes:
            if not code:
                continue
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 1000.0,
                    "amount": 10000.0,
                }
            )
        if api_name == "daily":
            return pd.DataFrame(rows, columns=DAILY_FIELDS)
        if api_name == "adj_factor":
            return pd.DataFrame(
                [{"ts_code": row["ts_code"], "trade_date": row["trade_date"], "adj_factor": 1.0} for row in rows],
                columns=["ts_code", "trade_date", "adj_factor"],
            )
        raise AssertionError(f"Unexpected API call: {api_name}")


class MissingAdjFactorClient(FakeTushareClient):
    missing_code = "600519.SH"

    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        if api_name != "adj_factor":
            return super().call(api_name, params=params, fields=fields)

        params = params or {}
        self.calls.append((api_name, params.copy(), fields))
        codes = str(params.get("ts_code", "")).split(",")
        rows = []
        for code in codes:
            if not code or code == self.missing_code:
                continue
            rows.append({"ts_code": code, "trade_date": "20240102", "adj_factor": 1.0})
        return pd.DataFrame(rows, columns=["ts_code", "trade_date", "adj_factor"])


class EmptyTushareClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        self.calls.append((api_name, (params or {}).copy(), fields))
        if api_name == "daily":
            return pd.DataFrame(columns=DAILY_FIELDS)
        if api_name == "adj_factor":
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        raise AssertionError(f"Unexpected API call: {api_name}")


class TradeCalClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        self.calls.append((api_name, (params or {}).copy(), fields))
        if api_name == "trade_cal":
            return pd.DataFrame(
                [
                    {"exchange": "SSE", "cal_date": "20240101", "is_open": 0, "pretrade_date": "20231229"},
                    {"exchange": "SSE", "cal_date": "20240102", "is_open": 1, "pretrade_date": "20231229"},
                    {"exchange": "SSE", "cal_date": "20240103", "is_open": 1, "pretrade_date": "20240102"},
                    {"exchange": "SSE", "cal_date": "20240106", "is_open": 0, "pretrade_date": "20240105"},
                ]
            )
        return super().call(api_name, params=params, fields=fields)


class FailingTradeCalClient(FakeTushareClient):
    def call(self, api_name: str, params: dict | None = None, fields: list[str] | str | None = None) -> pd.DataFrame:
        if api_name == "trade_cal":
            raise RuntimeError("calendar unavailable")
        return super().call(api_name, params=params, fields=fields)


def fake_a_trade_calendar(trade_days: list[str], pre_exception: Exception | None = None) -> SimpleNamespace:
    dates = sorted(pd.Timestamp(day).normalize() for day in trade_days)

    def is_trade_date(dtime: str) -> bool:
        return pd.Timestamp(dtime).normalize() in dates

    def get_pre_trade_date(dtime: str, cnt: int = 1) -> str:
        if pre_exception is not None:
            raise pre_exception
        target = pd.Timestamp(dtime).normalize()
        previous = [day for day in dates if day < target]
        return previous[-cnt].strftime("%Y-%m-%d")

    def get_next_trade_date(dtime: str, cnt: int = 1) -> str | None:
        target = pd.Timestamp(dtime).normalize()
        future = [day for day in dates if day > target]
        if len(future) < cnt:
            return None
        return future[cnt - 1].strftime("%Y-%m-%d")

    return SimpleNamespace(
        calendar_util=SimpleNamespace(
            _a_trade_cal_df=pd.DataFrame({"dt": [day.strftime("%Y-%m-%d") for day in dates]}),
            end_dt=dates[-1].strftime("%Y-%m-%d"),
        ),
        is_trade_date=is_trade_date,
        get_pre_trade_date=get_pre_trade_date,
        get_next_trade_date=get_next_trade_date,
    )


class DataFetcherTests(unittest.TestCase):
    def test_fetch_trade_calendar_normalizes_tushare_response(self) -> None:
        client = TradeCalClient()

        calendar = fetch_trade_calendar("2024-01-01", "2024-01-03", client=client)

        self.assertEqual(client.calls[0][0], "trade_cal")
        self.assertEqual(client.calls[0][1]["start_date"], "20240101")
        self.assertEqual(client.calls[0][1]["end_date"], "20240103")
        self.assertEqual(calendar["cal_date"].tolist()[:2], [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        self.assertEqual(calendar["is_open"].tolist()[:2], [0, 1])

    def test_required_latest_data_date_uses_previous_trade_day_before_cutoff(self) -> None:
        required = required_latest_data_date({}, now=pd.Timestamp("2024-01-03 19:59", tz="Asia/Shanghai"), client=TradeCalClient())
        self.assertEqual(required, pd.Timestamp("2024-01-02"))

    def test_required_latest_data_date_uses_same_trade_day_at_cutoff(self) -> None:
        required = required_latest_data_date({}, now=pd.Timestamp("2024-01-03 20:00", tz="Asia/Shanghai"), client=TradeCalClient())
        self.assertEqual(required, pd.Timestamp("2024-01-03"))

    def test_required_latest_data_date_uses_latest_open_day_on_non_trading_day(self) -> None:
        required = required_latest_data_date({}, now=pd.Timestamp("2024-01-06 21:00", tz="Asia/Shanghai"), client=TradeCalClient())
        self.assertEqual(required, pd.Timestamp("2024-01-03"))

    def test_required_latest_data_date_uses_a_trade_calendar_previous_trade_day_before_cutoff(self) -> None:
        calendar = fake_a_trade_calendar(["2024-01-02", "2024-01-03", "2024-01-05"])

        with patch.dict(sys.modules, {"a_trade_calendar": calendar}):
            required = required_latest_data_date({}, now=pd.Timestamp("2024-01-03 19:59", tz="Asia/Shanghai"), client=FailingTradeCalClient())

        self.assertEqual(required, pd.Timestamp("2024-01-02"))

    def test_required_latest_data_date_uses_a_trade_calendar_same_trade_day_at_cutoff(self) -> None:
        calendar = fake_a_trade_calendar(["2024-01-02", "2024-01-03", "2024-01-05"])

        with patch.dict(sys.modules, {"a_trade_calendar": calendar}):
            required = required_latest_data_date({}, now=pd.Timestamp("2024-01-03 20:00", tz="Asia/Shanghai"), client=FailingTradeCalClient())

        self.assertEqual(required, pd.Timestamp("2024-01-03"))

    def test_required_latest_data_date_uses_a_trade_calendar_latest_open_day_on_holiday(self) -> None:
        calendar = fake_a_trade_calendar(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-05"])

        with patch.dict(sys.modules, {"a_trade_calendar": calendar}):
            required = required_latest_data_date({}, now=pd.Timestamp("2024-01-01 21:00", tz="Asia/Shanghai"), client=FailingTradeCalClient())

        self.assertEqual(required, pd.Timestamp("2023-12-29"))

    def test_required_latest_data_date_falls_back_to_business_day_when_local_calendar_past_coverage(self) -> None:
        calendar = fake_a_trade_calendar(["2024-01-02", "2024-01-03", "2024-01-05"])

        with patch.dict(sys.modules, {"a_trade_calendar": calendar}):
            required = required_latest_data_date({}, now=pd.Timestamp("2024-01-08 21:00", tz="Asia/Shanghai"), client=FailingTradeCalClient())

        self.assertEqual(required, pd.Timestamp("2024-01-08"))

    def test_required_latest_data_date_falls_back_to_business_day_when_local_calendar_raises_index_error(self) -> None:
        calendar = fake_a_trade_calendar(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"], pre_exception=IndexError("out of bounds"))

        with patch.dict(sys.modules, {"a_trade_calendar": calendar}):
            required = required_latest_data_date({}, now=pd.Timestamp("2024-01-06 21:00", tz="Asia/Shanghai"), client=FailingTradeCalClient())

        self.assertEqual(required, pd.Timestamp("2024-01-05"))

    def test_required_latest_data_date_falls_back_to_business_day_when_local_calendar_missing(self) -> None:
        with patch("src.data_fetcher._required_latest_data_date_a_trade_calendar", side_effect=ImportError("no local calendar")):
            required = required_latest_data_date({}, now=pd.Timestamp("2024-01-06 21:00", tz="Asia/Shanghai"), client=FailingTradeCalClient())
        self.assertEqual(required, pd.Timestamp("2024-01-05"))

    def test_hs300_universe_uses_hs300_constituents_not_mainboard_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mainboard_file = root / "mainboard_a_stocks.csv"
            hs300_file = root / "hs300_constituents.csv"
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "name": "MAINBOARD_A"},
                    {"ts_code": "000002.SZ", "name": "MAINBOARD_B"},
                ]
            ).to_csv(mainboard_file, index=False)
            pd.DataFrame(
                [
                    {"con_code": "600000.SH"},
                    {"con_code": "000300.SZ"},
                ]
            ).to_csv(hs300_file, index=False)
            config = {
                "data": {
                    "universe": "hs300",
                    "constituents_file": str(mainboard_file),
                    "hs300_constituents_file": str(hs300_file),
                    "end_date": "2024-01-03",
                }
            }

            with patch("src.data_fetcher.load_config", return_value=config):
                codes = fetch_stock_universe()

            self.assertEqual(codes, ["000300.SZ", "600000.SH"])

    def test_filter_universe_frame_excludes_delisted_before_as_of_date(self) -> None:
        universe = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "PINGAN", "list_status": "L", "list_date": "19910403", "delist_date": ""},
                {"ts_code": "000003.SZ", "name": "DELISTED", "list_status": "D", "list_date": "19910403", "delist_date": "20020614"},
                {"ts_code": "000015.SZ", "name": "FUTURE_EXIT", "list_status": "D", "list_date": "19910403", "delist_date": "20251231"},
            ]
        )

        filtered = filter_universe_frame(universe, universe="mainboard_a", as_of_date="2024-01-01", exclude_st=True)

        self.assertEqual(filtered["ts_code"].tolist(), ["000001.SZ", "000015.SZ"])

    def test_filter_universe_frame_uses_point_in_time_st_calendar(self) -> None:
        universe = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "ST_STATIC_NAME", "list_status": "L", "list_date": "19910403", "delist_date": ""},
                {"ts_code": "000002.SZ", "name": "NORMAL", "list_status": "L", "list_date": "19910403", "delist_date": ""},
            ]
        )
        st_calendar = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "st_start_date": "20240601", "st_end_date": ""},
            ]
        )

        before = filter_universe_frame(
            universe,
            universe="mainboard_a",
            as_of_date="2024-05-31",
            exclude_st=True,
            st_calendar=st_calendar,
        )
        during = filter_universe_frame(
            universe,
            universe="mainboard_a",
            as_of_date="2024-06-01",
            exclude_st=True,
            st_calendar=st_calendar,
        )

        self.assertIn("000001.SZ", before["ts_code"].tolist())
        self.assertNotIn("000001.SZ", during["ts_code"].tolist())

    def test_fetch_daily_stocks_uses_comma_separated_batch_request(self) -> None:
        client = FakeTushareClient()

        df = fetch_daily_stocks(["000001.SZ", "600519.SH"], "2024-01-01", "2024-01-03", client=client)

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(len(daily_calls), 1)
        self.assertEqual(daily_calls[0][1]["ts_code"], "000001.SZ,600519.SH")
        self.assertEqual(sorted(df["ts_code"].unique().tolist()), ["000001.SZ", "600519.SH"])
        self.assertIn("adj_factor", df.columns)

    def test_fetch_daily_stocks_splits_long_range_into_date_windows(self) -> None:
        client = FakeTushareClient()

        fetch_daily_stocks(
            ["000001.SZ", "600519.SH"],
            "2024-01-01",
            "2024-01-05",
            client=client,
            window_days=2,
        )

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(len(daily_calls), 3)
        self.assertEqual([call[1]["start_date"] for call in daily_calls], ["20240101", "20240103", "20240105"])
        self.assertEqual([call[1]["end_date"] for call in daily_calls], ["20240102", "20240104", "20240105"])

    def test_update_daily_data_writes_each_symbol_from_batched_response(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "max_new_symbols_per_run": 100,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.TushareHttpClient.from_config", return_value=client
            ):
                written = update_daily_data(
                    stock_codes=["000001.SZ", "600519.SH"],
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

            self.assertEqual(set(written), {"000001.SZ", "600519.SH"})
            for code in written:
                path = raw_dir / f"{code}.csv"
                self.assertTrue(path.exists())
                df = pd.read_csv(path)
                self.assertEqual(df["ts_code"].tolist(), [code])
                self.assertIn("adj_factor", df.columns)

        daily_calls = [call for call in client.calls if call[0] == "daily"]
        self.assertEqual(len(daily_calls), 1)
        self.assertEqual(daily_calls[0][1]["ts_code"], "000001.SZ,600519.SH")

    def test_fetch_daily_stocks_skips_symbol_with_incomplete_adj_factor(self) -> None:
        client = MissingAdjFactorClient()

        df = fetch_daily_stocks(["000001.SZ", "600519.SH"], "2024-01-01", "2024-01-03", client=client)

        self.assertEqual(df["ts_code"].unique().tolist(), ["000001.SZ"])
        self.assertIn("adj_factor", df.columns)
        self.assertFalse(df["adj_factor"].isna().any())

    def test_update_daily_data_records_failed_symbol_when_adj_factor_is_incomplete(self) -> None:
        client = MissingAdjFactorClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "max_new_symbols_per_run": 100,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.TushareHttpClient.from_config", return_value=client
            ):
                written = update_daily_data(
                    stock_codes=["000001.SZ", "600519.SH"],
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

            self.assertEqual(set(written), {"000001.SZ"})
            self.assertTrue((raw_dir / "000001.SZ.csv").exists())
            self.assertFalse((raw_dir / "600519.SH.csv").exists())
            failed = pd.read_csv(raw_dir / "failed_fetches.csv")
            self.assertEqual(failed["ts_code"].tolist(), ["600519.SH"])
            self.assertEqual(failed["reason"].tolist(), ["empty_or_failed_fetch"])

    def test_update_daily_data_limits_new_symbol_backfill_but_keeps_existing_incremental_updates(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "max_new_symbols_per_run": 1,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            existing = raw_dir / "000001.SZ.csv"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-01",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(existing, index=False)

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.TushareHttpClient.from_config", return_value=client
            ):
                written = update_daily_data(
                    stock_codes=["000001.SZ", "600519.SH", "000002.SZ"],
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    raw_dir=raw_dir,
                )

            self.assertEqual(set(written), {"000001.SZ", "600519.SH"})
            self.assertTrue((raw_dir / "000001.SZ.csv").exists())
            self.assertTrue((raw_dir / "600519.SH.csv").exists())
            self.assertFalse((raw_dir / "000002.SZ.csv").exists())

    def test_resumable_update_prioritizes_missing_symbols_and_writes_progress(self) -> None:
        client = FakeTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 1,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "2024-01-03",
                        "open": 9.0,
                        "high": 9.0,
                        "low": 9.0,
                        "close": 9.0,
                        "vol": 100.0,
                        "amount": 900.0,
                        "adj_factor": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "000001.SZ.csv", index=False)

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.TushareHttpClient.from_config", return_value=client):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH", "000002.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=1,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(set(written), {"600519.SH"})
            self.assertTrue((raw_dir / "600519.SH.csv").exists())
            self.assertFalse((raw_dir / "000002.SZ.csv").exists())
            self.assertEqual(int(progress["initial_existing"]), 1)
            self.assertEqual(int(progress["pending_symbols"]), 2)
            self.assertEqual(int(progress["completed_symbols"]), 1)
            self.assertEqual(int(progress["remaining_symbols"]), 1)

    def test_resumable_update_marks_error_when_symbol_is_not_written(self) -> None:
        client = EmptyTushareClient()
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "daily_batch_size": 100,
                "update_chunk_size": 1,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.TushareHttpClient.from_config", return_value=client):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=1,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(written, {})
            self.assertEqual(progress["status"], "error")
            self.assertEqual(int(progress["failed_symbols"]), 1)
            self.assertIn("not_written", progress["last_error"])

    def test_resumable_update_calls_update_daily_data_once_per_chunk_start_group(self) -> None:
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "update_chunk_size": 3,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }
        calls: list[list[str]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"

            def fake_update_daily_data(stock_codes, start_date=None, end_date=None, raw_dir=None):
                codes = list(stock_codes)
                calls.append(codes)
                written = {}
                for code in codes:
                    path = Path(raw_dir) / f"{code}.csv"
                    path.write_text("", encoding="utf-8")
                    written[code] = path
                return written

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data", side_effect=fake_update_daily_data):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH", "000002.SZ"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=3,
                    sleep_seconds=0,
                    max_chunks=1,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(calls, [["000001.SZ", "600519.SH", "000002.SZ"]])
            self.assertEqual(set(written), {"000001.SZ", "600519.SH", "000002.SZ"})
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(int(progress["completed_symbols"]), 3)

    def test_resumable_update_include_existing_tracks_processed_symbols(self) -> None:
        config = {
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "raw_dir": "unused",
                "update_chunk_size": 1,
                "update_sleep_seconds": 0,
            },
            "tushare": {"http_url": "http://example.test", "token": "", "timeout": 30},
        }
        calls: list[list[str]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            progress_file = root / "progress.json"
            for code in ["000001.SZ", "600519.SH"]:
                (raw_dir / f"{code}.csv").write_text("", encoding="utf-8")

            def fake_update_daily_data(stock_codes, start_date=None, end_date=None, raw_dir=None):
                codes = list(stock_codes)
                calls.append(codes)
                return {code: Path(raw_dir) / f"{code}.csv" for code in codes}

            with patch("src.data_fetcher.load_config", return_value=config), patch(
                "src.data_fetcher.resolve_path", side_effect=lambda value: Path(value)
            ), patch("src.data_fetcher.update_daily_data", side_effect=fake_update_daily_data):
                written = update_daily_data_resumable(
                    stock_codes=["000001.SZ", "600519.SH"],
                    raw_dir=raw_dir,
                    progress_file=progress_file,
                    chunk_size=1,
                    sleep_seconds=0,
                    max_chunks=1,
                    include_existing=True,
                )

            progress = pd.read_json(progress_file, typ="series")

            self.assertEqual(calls, [["000001.SZ"]])
            self.assertEqual(set(written), {"000001.SZ"})
            self.assertEqual(progress["status"], "partial")
            self.assertEqual(int(progress["completed_symbols"]), 1)
            self.assertEqual(int(progress["remaining_symbols"]), 1)


if __name__ == "__main__":
    unittest.main()
