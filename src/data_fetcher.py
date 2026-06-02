from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from urllib.parse import urlparse
from pathlib import Path
import random
import time
from typing import Iterable

import pandas as pd

from src.config_loader import load_config, resolve_path


logger = logging.getLogger(__name__)

DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
]
ADJ_FACTOR_FIELDS = ["ts_code", "trade_date", "adj_factor"]
TRADE_CAL_FIELDS = ["exchange", "cal_date", "is_open", "pretrade_date"]
STOCK_BASIC_FIELDS = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
]
MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")


def _format_tushare_date(value: str | datetime | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.strftime("%Y%m%d")


def _parse_tushare_frame(data: dict) -> pd.DataFrame:
    payload = data.get("data", data)
    fields = payload.get("fields")
    items = payload.get("items")
    if fields is None or items is None:
        raise ValueError(f"Unexpected tushare response shape: {data}")
    return pd.DataFrame(items, columns=fields)


@dataclass
class TushareHttpClient:
    http_url: str
    token: str | None = None
    timeout: int = 30

    @classmethod
    def from_config(cls, config: dict | None = None) -> "TushareHttpClient":
        cfg = config or load_config()
        ts_cfg = cfg.get("tushare", {})
        return cls(
            http_url=ts_cfg.get("http_url", ""),
            token=ts_cfg.get("token") or None,
            timeout=int(ts_cfg.get("timeout", 30)),
        )

    def call(self, api_name: str, params: dict | None = None, fields: Iterable[str] | str | None = None) -> pd.DataFrame:
        if not self.http_url or "your-proxy-server" in self.http_url:
            raise RuntimeError("Please configure tushare.http_url in config/settings.yaml first.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("The 'requests' package is required for tushare HTTP access.") from exc

        if isinstance(fields, str):
            field_value = fields
        elif fields is None:
            field_value = None
        else:
            field_value = ",".join(fields)

        payload = {
            "api_name": api_name,
            "token": None if self.token == "your_token" else self.token,
            "params": params or {},
        }
        if field_value:
            payload["fields"] = field_value

        try:
            response = requests.post(self.http_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            endpoint = describe_endpoint(self.http_url)
            raise RuntimeError(
                "Failed to connect to tushare HTTP proxy "
                f"({endpoint}). Check the full proxy URL/path, firewall/network permission, "
                "and whether the proxy service is running."
            ) from exc
        except ValueError as exc:
            raise RuntimeError("The tushare HTTP proxy returned a non-JSON response.") from exc
        if data.get("code", 0) != 0:
            raise RuntimeError(f"tushare error: {data.get('msg', data)}")
        return _parse_tushare_frame(data)

    def redacted_request_preview(
        self,
        api_name: str = "daily",
        params: dict | None = None,
        fields: Iterable[str] | str | None = None,
    ) -> dict:
        if isinstance(fields, str):
            field_value = fields
        elif fields is None:
            field_value = None
        else:
            field_value = ",".join(fields)
        payload = {
            "api_name": api_name,
            "token": "***" if self.token else None,
            "params": params or {},
        }
        if field_value:
            payload["fields"] = field_value
        return {"url": describe_endpoint(self.http_url), "timeout": self.timeout, "payload": payload}


def fetch_trade_calendar(
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient | None = None,
    exchange: str = "SSE",
) -> pd.DataFrame:
    client = client or TushareHttpClient.from_config()
    params = {
        "exchange": exchange,
        "start_date": _format_tushare_date(start_date),
        "end_date": _format_tushare_date(end_date),
    }
    df = client.call("trade_cal", params=params, fields=TRADE_CAL_FIELDS)
    if df.empty:
        return pd.DataFrame(columns=TRADE_CAL_FIELDS)
    result = df.copy()
    result["cal_date"] = pd.to_datetime(result["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    if "pretrade_date" in result.columns:
        result["pretrade_date"] = pd.to_datetime(result["pretrade_date"].astype(str), format="%Y%m%d", errors="coerce")
    result["is_open"] = pd.to_numeric(result["is_open"], errors="coerce").fillna(0).astype(int)
    return result.dropna(subset=["cal_date"]).sort_values("cal_date").reset_index(drop=True)


def required_latest_data_date(
    data_cfg: dict,
    now: datetime | pd.Timestamp | None = None,
    client: TushareHttpClient | None = None,
) -> pd.Timestamp:
    """Return the minimum acceptable data date for a latest signal.

    Tushare daily data is usually available in the evening after market close.
    Before the cutoff hour on an open trading day, require data through the
    previous trading day; at/after the cutoff, require data through that trading
    day. On non-trading days, require the latest open trading day before today.
    Falls back to business-day approximation if the Tushare calendar is not
    available, so offline workflows still get a conservative check.
    """
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Shanghai")
    if now_ts.tzinfo is not None:
        now_local = now_ts.tz_convert("Asia/Shanghai")
    else:
        now_local = now_ts.tz_localize("Asia/Shanghai")

    cutoff_hour = int(data_cfg.get("latest_data_cutoff_hour", 20))
    today = now_local.normalize().tz_localize(None)
    calendar_lookback_days = int(data_cfg.get("trade_calendar_lookback_days", 45))
    try:
        calendar = fetch_trade_calendar(
            today - pd.Timedelta(days=calendar_lookback_days),
            today,
            client=client,
            exchange=str(data_cfg.get("trade_calendar_exchange", "SSE")),
        )
        required = _required_latest_data_date_from_calendar(calendar, today, now_local.hour, cutoff_hour)
        if required is not None:
            return required
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Falling back to local A-share calendar because trade_cal failed: %s", exc)

    try:
        return _required_latest_data_date_a_trade_calendar(data_cfg, now=now_local)
    except (ImportError, AttributeError, ValueError, TypeError) as exc:
        logger.warning("Falling back to business-day latest data date because a-trade-calendar failed: %s", exc)

    return _required_latest_data_date_business_day(data_cfg, now=now_local)


def _required_latest_data_date_from_calendar(
    calendar: pd.DataFrame,
    today: pd.Timestamp,
    current_hour: int,
    cutoff_hour: int,
) -> pd.Timestamp | None:
    if calendar.empty or "cal_date" not in calendar.columns or "is_open" not in calendar.columns:
        return None
    cal = calendar.copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"]).dt.normalize()
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    open_days = pd.DatetimeIndex(cal.loc[(cal["is_open"] == 1) & (cal["cal_date"] <= today), "cal_date"]).sort_values().unique()
    if open_days.empty:
        return None
    latest_open = pd.Timestamp(open_days[-1]).normalize()
    today_is_open = latest_open == today
    if today_is_open and current_hour < cutoff_hour:
        previous_open_days = open_days[open_days < today]
        if previous_open_days.empty:
            return None
        return pd.Timestamp(previous_open_days[-1]).normalize()
    return latest_open


def _required_latest_data_date_a_trade_calendar(data_cfg: dict, now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    try:
        import a_trade_calendar
    except ImportError:
        raise

    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Shanghai")
    if now_ts.tzinfo is not None:
        now_local = now_ts.tz_convert("Asia/Shanghai")
    else:
        now_local = now_ts.tz_localize("Asia/Shanghai")

    cutoff_hour = int(data_cfg.get("latest_data_cutoff_hour", 20))
    today = now_local.normalize().tz_localize(None)
    today_text = today.strftime("%Y-%m-%d")
    if a_trade_calendar.is_trade_date(today_text):
        if now_local.hour < cutoff_hour:
            return pd.Timestamp(a_trade_calendar.get_pre_trade_date(today_text, 1)).normalize()
        return today
    latest = a_trade_calendar.get_pre_trade_date(today_text, 1)
    return pd.Timestamp(latest).normalize()


def _required_latest_data_date_business_day(data_cfg: dict, now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Shanghai")
    if now_ts.tzinfo is not None:
        now_local = now_ts.tz_convert("Asia/Shanghai")
    else:
        now_local = now_ts.tz_localize("Asia/Shanghai")
    cutoff_hour = int(data_cfg.get("latest_data_cutoff_hour", 20))
    today = now_local.normalize().tz_localize(None)
    latest_trading_day = _previous_or_same_business_day(today)
    if today == latest_trading_day and now_local.hour < cutoff_hour:
        return _previous_business_day(latest_trading_day)
    return latest_trading_day


def _previous_or_same_business_day(date: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(date).normalize()
    while date.weekday() >= 5:
        date -= pd.Timedelta(days=1)
    return date


def _previous_business_day(date: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(date).normalize() - pd.Timedelta(days=1)
    return _previous_or_same_business_day(date)


def fetch_daily_stock(
    ts_code: str,
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    client = client or TushareHttpClient.from_config()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            params = {
                "ts_code": ts_code,
                "start_date": _format_tushare_date(start_date),
                "end_date": _format_tushare_date(end_date),
            }
            df = client.call("daily", params=params, fields=DAILY_FIELDS)
            daily = normalize_daily_frame(df, default_ts_code=ts_code)
            adj = client.call("adj_factor", params=params, fields=ADJ_FACTOR_FIELDS)
            return merge_adj_factor(daily, adj, default_ts_code=ts_code)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1) + random.uniform(0, 1)
                logger.warning("Retrying %s daily data after error: %s", ts_code, exc)
                time.sleep(wait_seconds)
    raise ValueError(f"{ts_code} daily data response is invalid after {retries} attempts: {last_error}") from last_error


def fetch_daily_stocks(
    ts_codes: Iterable[str],
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient | None = None,
    retries: int = 3,
    batch_size: int = 100,
    window_days: int | None = None,
    skip_failed: bool = True,
) -> pd.DataFrame:
    codes = [str(code) for code in dict.fromkeys(ts_codes)]
    if not codes:
        return pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
    if len(codes) == 1 or batch_size <= 1:
        frames = []
        failed_codes: list[str] = []
        for code in codes:
            try:
                frames.append(fetch_daily_stock(code, start_date, end_date, client=client, retries=retries))
            except (RuntimeError, ValueError) as exc:
                if not skip_failed:
                    raise
                failed_codes.append(code)
                logger.error("Skipping %s after daily fetch failure: %s", code, exc)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
        result.attrs["failed_codes"] = failed_codes
        return result

    client = client or TushareHttpClient.from_config()
    frames: list[pd.DataFrame] = []
    all_failed_codes: set[str] = set()
    for batch in _batched(codes, batch_size):
        batch_frames: list[pd.DataFrame] = []
        failed_codes: set[str] = set()
        for window_start, window_end in _date_windows(start_date, end_date, window_days):
            window_df = _fetch_daily_stock_batch(
                batch,
                window_start,
                window_end,
                client=client,
                retries=retries,
                skip_failed=skip_failed,
            )
            failed_codes.update(window_df.attrs.get("failed_codes", []))
            batch_frames.append(window_df)
        if batch_frames:
            batch_df = pd.concat(batch_frames, ignore_index=True)
            if failed_codes:
                batch_df = batch_df[~batch_df["ts_code"].isin(failed_codes)].copy() if not batch_df.empty else batch_df
            all_failed_codes.update(failed_codes)
            frames.append(batch_df)
    if not frames:
        result = pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
        result.attrs["failed_codes"] = sorted(all_failed_codes)
        return result
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["ts_code", "trade_date"], keep="last")
    result.attrs["failed_codes"] = sorted(all_failed_codes)
    return result


def _fetch_daily_stock_batch(
    ts_codes: list[str],
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient,
    retries: int = 3,
    skip_failed: bool = True,
) -> pd.DataFrame:
    params = {
        "ts_code": ",".join(ts_codes),
        "start_date": _format_tushare_date(start_date),
        "end_date": _format_tushare_date(end_date),
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = client.call("daily", params=params, fields=DAILY_FIELDS)
            daily = normalize_daily_frame(df)
            adj = _fetch_adj_factor_batch(ts_codes, start_date, end_date, client=client)
            adj = _complete_missing_adj_factors(daily, adj, start_date, end_date, client=client, retries=retries, skip_failed=skip_failed)
            if skip_failed:
                daily, adj, failed_codes = _drop_incomplete_adj_symbols(daily, adj)
                if failed_codes:
                    logger.error(
                        "Skipping %d symbols with incomplete adj_factor coverage in batch: %s",
                        len(failed_codes),
                        ",".join(failed_codes[:10]),
                    )
                if daily.empty:
                    result = pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
                    result.attrs["failed_codes"] = failed_codes
                    return result
            result = merge_adj_factor(daily, adj)
            result.attrs["failed_codes"] = failed_codes if skip_failed else []
            return result
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if skip_failed and _is_tushare_connection_error(exc):
                break
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1) + random.uniform(0, 1)
                logger.warning("Retrying %d-stock daily batch after error: %s", len(ts_codes), exc)
                time.sleep(wait_seconds)
    logger.warning("Falling back to per-stock daily fetch for %d symbols after batch error: %s", len(ts_codes), last_error)
    frames = []
    failed_codes: list[str] = []
    fallback_retries = 1 if last_error is not None and _is_tushare_connection_error(last_error) else retries
    for code in ts_codes:
        try:
            frames.append(fetch_daily_stock(code, start_date, end_date, client=client, retries=fallback_retries))
        except (RuntimeError, ValueError) as exc:
            if not skip_failed:
                raise
            failed_codes.append(code)
            logger.error("Skipping %s after per-stock fallback failure: %s", code, exc)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
    result.attrs["failed_codes"] = failed_codes
    return result


def _is_tushare_connection_error(exc: Exception) -> bool:
    return "Failed to connect to tushare HTTP proxy" in str(exc)


def _fetch_adj_factor_batch(
    ts_codes: list[str],
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient,
) -> pd.DataFrame:
    params = {
        "ts_code": ",".join(ts_codes),
        "start_date": _format_tushare_date(start_date),
        "end_date": _format_tushare_date(end_date),
    }
    try:
        adj = client.call("adj_factor", params=params, fields=ADJ_FACTOR_FIELDS)
        if not adj.empty:
            return adj
    except (RuntimeError, ValueError) as exc:
        logger.warning("Falling back to per-stock adj_factor fetch for %d symbols: %s", len(ts_codes), exc)

    frames = []
    for code in ts_codes:
        per_stock_params = {**params, "ts_code": code}
        adj = client.call("adj_factor", params=per_stock_params, fields=ADJ_FACTOR_FIELDS)
        if not adj.empty:
            frames.append(adj)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ADJ_FACTOR_FIELDS)


def _complete_missing_adj_factors(
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    client: TushareHttpClient,
    retries: int,
    skip_failed: bool,
) -> pd.DataFrame:
    if daily.empty:
        return adj
    daily_norm = normalize_daily_frame(daily)
    adj_norm = _normalize_adj_factor_frame(adj)
    expected = daily_norm[["ts_code", "trade_date"]].drop_duplicates()
    available = adj_norm[["ts_code", "trade_date", "adj_factor"]].drop_duplicates(["ts_code", "trade_date"])
    coverage = expected.merge(available, on=["ts_code", "trade_date"], how="left")
    missing_codes = sorted(coverage.loc[coverage["adj_factor"].isna(), "ts_code"].dropna().astype(str).unique())
    if not missing_codes:
        return adj_norm

    logger.warning(
        "Fetching incomplete adj_factor coverage for %d/%d symbols.",
        len(missing_codes),
        daily_norm["ts_code"].nunique(),
    )
    frames = [adj_norm] if not adj_norm.empty else []
    for code in missing_codes:
        try:
            params = {
                "ts_code": code,
                "start_date": _format_tushare_date(start_date),
                "end_date": _format_tushare_date(end_date),
            }
            piece = client.call("adj_factor", params=params, fields=ADJ_FACTOR_FIELDS)
            if not piece.empty:
                frames.append(_normalize_adj_factor_frame(piece, default_ts_code=code))
        except (RuntimeError, ValueError) as exc:
            if not skip_failed:
                raise
            logger.error("Skipping %s adj_factor completion after failure: %s", code, exc)
    if not frames:
        return pd.DataFrame(columns=ADJ_FACTOR_FIELDS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")


def _drop_incomplete_adj_symbols(daily: pd.DataFrame, adj_factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    daily_norm = normalize_daily_frame(daily)
    adj_norm = _normalize_adj_factor_frame(adj_factor)
    if daily_norm.empty:
        return daily_norm, adj_norm, []

    expected = daily_norm[["ts_code", "trade_date"]].drop_duplicates()
    available = adj_norm[["ts_code", "trade_date", "adj_factor"]].drop_duplicates(["ts_code", "trade_date"])
    coverage = expected.merge(available, on=["ts_code", "trade_date"], how="left")
    failed_codes = sorted(coverage.loc[coverage["adj_factor"].isna(), "ts_code"].dropna().astype(str).unique())
    if not failed_codes:
        return daily_norm, adj_norm, []

    failed_set = set(failed_codes)
    daily_clean = daily_norm[~daily_norm["ts_code"].isin(failed_set)].copy()
    adj_clean = adj_norm[~adj_norm["ts_code"].isin(failed_set)].copy()
    return daily_clean, adj_clean, failed_codes


def _normalize_adj_factor_frame(adj_factor: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    adj = adj_factor.rename(columns={"date": "trade_date"}).copy()
    if "ts_code" not in adj.columns and default_ts_code:
        adj["ts_code"] = default_ts_code
    required = set(ADJ_FACTOR_FIELDS)
    if not required.issubset(adj.columns):
        if adj.empty:
            return pd.DataFrame(columns=ADJ_FACTOR_FIELDS)
        missing = sorted(required - set(adj.columns))
        raise ValueError(f"Adj factor data is missing columns: {missing}")
    adj = adj[ADJ_FACTOR_FIELDS]
    adj["trade_date"] = pd.to_datetime(adj["trade_date"].astype(str), errors="coerce")
    adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
    adj = adj.dropna(subset=["trade_date", "adj_factor"])
    return adj.drop_duplicates(["ts_code", "trade_date"], keep="last").reset_index(drop=True)


def merge_adj_factor(daily: pd.DataFrame, adj_factor: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    daily = normalize_daily_frame(daily, default_ts_code=default_ts_code)
    if daily.empty:
        result = daily.copy()
        result["adj_factor"] = pd.Series(dtype="float64")
        return result
    adj = _normalize_adj_factor_frame(adj_factor, default_ts_code=default_ts_code)
    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    if merged["adj_factor"].isna().any():
        missing_count = int(merged["adj_factor"].isna().sum())
        raise ValueError(f"Missing adj_factor for {missing_count} daily rows.")
    return merged


def fetch_hs300_stocks(
    date: str | datetime | None = None,
    client: TushareHttpClient | None = None,
    local_file: str | Path | None = None,
) -> list[str]:
    config = load_config()
    data_cfg = config.get("data", {})
    local_path = resolve_path(local_file or data_cfg.get("hs300_constituents_file", "data/raw/hs300_constituents.csv"))
    if local_path.exists():
        df = pd.read_csv(local_path)
        code_col = "ts_code" if "ts_code" in df.columns else "con_code"
        if code_col not in df.columns:
            raise ValueError(f"{local_path} must contain ts_code or con_code column.")
        return sorted(df[code_col].dropna().astype(str).unique().tolist())

    client = client or TushareHttpClient.from_config(config)
    params = {"index_code": "000300.SH"}
    if date is not None:
        params["trade_date"] = _format_tushare_date(date)
    df = client.call(
        "index_weight",
        params=params,
        fields=["index_code", "con_code", "trade_date", "weight"],
    )
    if df.empty:
        raise RuntimeError("No HS300 constituents returned. Add data/raw/hs300_constituents.csv or check proxy params.")
    return sorted(df["con_code"].dropna().astype(str).unique().tolist())


def fetch_stock_universe(
    universe: str | None = None,
    date: str | datetime | None = None,
    client: TushareHttpClient | None = None,
    local_file: str | Path | None = None,
    save_metadata: bool = True,
) -> list[str]:
    config = load_config()
    data_cfg = config.get("data", {})
    universe = (universe or data_cfg.get("universe", "mainboard_a")).lower()

    if universe in {"hs300", "csi300"}:
        return fetch_hs300_stocks(date=date, client=client, local_file=local_file)

    local_path = resolve_path(local_file or data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if local_path.exists():
        df = pd.read_csv(local_path)
    else:
        client = client or TushareHttpClient.from_config(config)
        df = _fetch_stock_basic_history(client)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if save_metadata:
            df.to_csv(local_path, index=False, encoding="utf-8-sig")

    filtered = filter_universe_frame(
        df,
        universe=universe,
        as_of_date=date or data_cfg.get("end_date"),
        exclude_st=bool(data_cfg.get("exclude_st", True)),
        st_calendar=_load_st_calendar(data_cfg.get("st_calendar_file")),
    )
    code_col = _code_column(filtered)
    return sorted(filtered[code_col].dropna().astype(str).unique().tolist())


def _fetch_stock_basic_history(client: TushareHttpClient) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for status in ["L", "D"]:
        df = client.call(
            "stock_basic",
            params={"list_status": status},
            fields=STOCK_BASIC_FIELDS,
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No stock_basic rows returned for A-share universe.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")


def filter_universe_frame(
    df: pd.DataFrame,
    universe: str,
    as_of_date: str | datetime | None = None,
    exclude_st: bool = True,
    st_calendar: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    code_col = _code_column(result)
    result[code_col] = result[code_col].astype(str).str.upper()

    if universe in {"mainboard_a", "a_mainboard", "mainboard"}:
        result = result[result[code_col].map(_is_mainboard_code)]
    elif universe in {"all_a", "a_share", "ashare"}:
        result = result[result[code_col].str.endswith((".SH", ".SZ"))]
    elif universe in {"hs300", "csi300"}:
        return result
    else:
        raise ValueError(f"Unsupported universe: {universe}")

    if exclude_st and st_calendar is not None and as_of_date is not None:
        result = _exclude_point_in_time_st(result, code_col, as_of_date, st_calendar)
    elif exclude_st and "name" in result.columns:
        names = result["name"].fillna("").astype(str).str.upper()
        result = result[~names.str.contains("ST", regex=False)]

    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        if "list_date" in result.columns:
            listed = pd.to_datetime(result["list_date"].astype(str), format="%Y%m%d", errors="coerce")
            result = result[listed.isna() | (listed <= as_of)]
        if "delist_date" in result.columns:
            delisted = pd.to_datetime(result["delist_date"].astype(str), format="%Y%m%d", errors="coerce")
            result = result[delisted.isna() | (delisted > as_of)]
        if "list_status" in result.columns:
            status = result["list_status"].fillna("L").astype(str).str.upper()
            if "delist_date" in result.columns:
                delisted = pd.to_datetime(result["delist_date"].astype(str), format="%Y%m%d", errors="coerce")
                result = result[(status == "L") | (delisted.notna() & (delisted > as_of))]
            else:
                result = result[status == "L"]
    return result


def _load_st_calendar(path_value: str | Path | None) -> pd.DataFrame | None:
    if not path_value:
        return None
    path = resolve_path(path_value)
    if not path.exists():
        logger.warning("Configured ST calendar file does not exist: %s", path)
        return None
    return pd.read_csv(path)


def _exclude_point_in_time_st(
    df: pd.DataFrame,
    code_col: str,
    as_of_date: str | datetime,
    st_calendar: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    calendar = pd.read_csv(resolve_path(st_calendar)) if isinstance(st_calendar, (str, Path)) else st_calendar.copy()
    if calendar.empty:
        return df
    st_code_col = _code_column(calendar)
    start_col = next((col for col in ["st_start_date", "start_date", "begin_date", "date"] if col in calendar.columns), None)
    end_col = next((col for col in ["st_end_date", "end_date", "remove_date"] if col in calendar.columns), None)
    if start_col is None:
        raise ValueError("ST calendar must contain one of: st_start_date, start_date, begin_date, date.")

    as_of = pd.Timestamp(as_of_date)
    calendar[st_code_col] = calendar[st_code_col].astype(str).str.upper()
    starts = _parse_calendar_dates(calendar[start_col])
    ends = _parse_calendar_dates(calendar[end_col]) if end_col is not None else pd.Series(pd.NaT, index=calendar.index)
    active = (starts <= as_of) & (ends.isna() | (ends >= as_of))
    st_codes = set(calendar.loc[active, st_code_col].dropna().astype(str).str.upper())
    return df[~df[code_col].astype(str).str.upper().isin(st_codes)]


def _parse_calendar_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _code_column(df: pd.DataFrame) -> str:
    for col in ["ts_code", "con_code", "instrument", "code"]:
        if col in df.columns:
            return col
    raise ValueError("Universe file must contain one of: ts_code, con_code, instrument, code.")


def _is_mainboard_code(code: str) -> bool:
    symbol = code.split(".", 1)[0]
    exchange_ok = code.endswith((".SH", ".SZ"))
    return exchange_ok and symbol.startswith(MAINBOARD_PREFIXES)


def normalize_daily_frame(df: pd.DataFrame, default_ts_code: str | None = None) -> pd.DataFrame:
    renamed = df.rename(columns={"volume": "vol", "date": "trade_date"}).copy()
    if "ts_code" not in renamed.columns and default_ts_code:
        renamed["ts_code"] = default_ts_code
    missing = [col for col in DAILY_FIELDS if col not in renamed.columns]
    if missing:
        if renamed.empty:
            return pd.DataFrame(columns=DAILY_FIELDS)
        raise ValueError(f"Daily data is missing columns: {missing}")

    renamed = renamed[DAILY_FIELDS]
    renamed["trade_date"] = pd.to_datetime(renamed["trade_date"].astype(str), errors="coerce")
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    if "adj_factor" in df.columns:
        renamed["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    renamed = renamed.dropna(subset=["trade_date", "close"]).sort_values(["ts_code", "trade_date"])
    return renamed.reset_index(drop=True)


def update_daily_data(
    stock_codes: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_dir: str | Path | None = None,
) -> dict[str, Path]:
    config = load_config()
    data_cfg = config.get("data", {})
    duplicate_keep = str(data_cfg.get("duplicate_keep", "first"))
    target_dir = resolve_path(raw_dir or data_cfg.get("raw_dir", "data/raw"))
    target_dir.mkdir(parents=True, exist_ok=True)

    start = start_date or data_cfg["start_date"]
    end = end_date or data_cfg["end_date"]
    codes = list(stock_codes) if stock_codes is not None else fetch_stock_universe()
    client = TushareHttpClient.from_config(config)
    batch_size = int(data_cfg.get("daily_batch_size", 100))
    window_days = int(data_cfg.get("daily_window_days", 500))
    max_new_symbols = data_cfg.get("max_new_symbols_per_run")
    max_new_symbols = int(max_new_symbols) if max_new_symbols is not None else None
    retries = int(data_cfg.get("retries", 3))

    written: dict[str, Path] = {}
    failed: dict[str, str] = {}
    pending: dict[str, tuple[Path, str, bool]] = {}
    for code in codes:
        path = target_dir / f"{code}.csv"
        needs_adj_backfill = _needs_adj_factor_backfill(path)
        actual_start = start if needs_adj_backfill else _incremental_start(path, start)
        if pd.Timestamp(actual_start) > pd.Timestamp(end):
            written[code] = path
            continue
        pending[code] = (path, actual_start, needs_adj_backfill)
    pending = _limit_new_symbols_per_run(pending, max_new_symbols=max_new_symbols)
    if pending:
        logger.info("Updating %d pending stock files.", len(pending))

    for actual_start, grouped_codes in _group_codes_by_start(pending).items():
        for batch_codes in _batched(grouped_codes, batch_size):
            batch_df = fetch_daily_stocks(
                batch_codes,
                actual_start,
                end,
                client=client,
                retries=retries,
                batch_size=batch_size,
                window_days=window_days,
                skip_failed=True,
            )
            batch_failed = set(batch_df.attrs.get("failed_codes", []))
            for code in batch_codes:
                path, _actual_start, needs_adj_backfill = pending[code]
                if code in batch_failed:
                    failed[code] = "empty_or_failed_fetch"
                    continue
                if batch_df.empty:
                    new_df = pd.DataFrame(columns=[*DAILY_FIELDS, "adj_factor"])
                else:
                    new_df = batch_df[batch_df["ts_code"] == code].copy()
                if new_df.empty and path.exists():
                    written[code] = path
                    continue
                if new_df.empty:
                    failed[code] = "empty_or_failed_fetch"
                    continue
                if path.exists() and not needs_adj_backfill:
                    old_df = pd.read_csv(path, parse_dates=["trade_date"])
                    new_df = pd.concat([old_df, new_df], ignore_index=True)
                new_df = normalize_daily_frame(new_df, default_ts_code=code)
                if "adj_factor" in new_df.columns:
                    new_df = new_df.dropna(subset=["adj_factor"])
                new_df = new_df.drop_duplicates(["ts_code", "trade_date"], keep=duplicate_keep)
                new_df.to_csv(path, index=False, encoding="utf-8-sig")
                written[code] = path
            logger.info("Updated %d/%d pending stock files.", len(written), len(pending))
    if failed:
        failed_path = target_dir / "failed_fetches.csv"
        pd.DataFrame({"ts_code": list(failed), "reason": list(failed.values())}).to_csv(
            failed_path, index=False, encoding="utf-8-sig"
        )
        logger.warning("Skipped %d symbols during data update. See %s", len(failed), failed_path)
    return written


def update_daily_data_resumable(
    stock_codes: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_dir: str | Path | None = None,
    chunk_size: int | None = None,
    sleep_seconds: float | None = None,
    progress_file: str | Path | None = None,
    max_chunks: int | None = None,
    include_existing: bool = False,
) -> dict[str, Path]:
    config = load_config()
    data_cfg = config.get("data", {})
    target_dir = resolve_path(raw_dir or data_cfg.get("raw_dir", "data/raw"))
    target_dir.mkdir(parents=True, exist_ok=True)

    start = start_date or data_cfg["start_date"]
    end = end_date or data_cfg["end_date"]
    codes = list(stock_codes) if stock_codes is not None else fetch_stock_universe()
    codes = [str(code).upper() for code in dict.fromkeys(codes)]
    chunk_size = max(1, int(chunk_size or data_cfg.get("update_chunk_size", 20)))
    sleep_seconds = float(sleep_seconds if sleep_seconds is not None else data_cfg.get("update_sleep_seconds", 90))
    progress_path = resolve_path(progress_file or data_cfg.get("update_progress_file", "outputs/data_update_progress.json"))

    initial_existing = _existing_stock_codes(target_dir)
    pending_codes = codes if include_existing else [code for code in codes if code not in initial_existing]
    list_dates = _load_universe_list_dates(config)
    written: dict[str, Path] = {}
    failed: dict[str, str] = {}
    processed_codes: set[str] = set()
    last_error = ""
    started_at = datetime.now().isoformat(timespec="seconds")
    logger.info(
        "Resumable update: %d target symbols, %d existing raw files, %d symbols pending.",
        len(codes),
        len(initial_existing & set(codes)),
        len(pending_codes),
    )
    _write_update_progress(
        progress_path,
        {
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "target_symbols": len(codes),
            "initial_existing": len(initial_existing & set(codes)),
            "pending_symbols": len(pending_codes),
            "chunk_size": chunk_size,
            "sleep_seconds": sleep_seconds,
            "completed_symbols": 0,
            "failed_symbols": 0,
            "remaining_symbols": len(pending_codes),
            "last_chunk": [],
            "current_symbol": "",
            "last_error": "",
        },
    )

    chunks_run = 0
    for batch_codes in _batched(pending_codes, chunk_size):
        if max_chunks is not None and chunks_run >= max_chunks:
            break
        chunks_run += 1
        chunk_error = ""
        logger.info(
            "Updating missing-symbol chunk %d: %s",
            chunks_run,
            ",".join(batch_codes[:5]) + ("..." if len(batch_codes) > 5 else ""),
        )
        chunk_starts = {code: _symbol_start_date(start, list_dates.get(code)) for code in batch_codes}
        for code_start, grouped_codes in _group_chunk_codes_by_start(chunk_starts).items():
            try:
                batch_written = update_daily_data(
                    stock_codes=grouped_codes,
                    start_date=code_start,
                    end_date=end,
                    raw_dir=target_dir,
                )
                written.update(batch_written)
            except Exception as exc:
                chunk_error = str(exc)
                last_error = chunk_error
                for code in grouped_codes:
                    failed[code] = chunk_error
                logger.error("Code group failed in chunk %d (%s): %s", chunks_run, ",".join(grouped_codes[:5]), exc)
            finally:
                processed_codes.update(grouped_codes)

            existing_now = _existing_stock_codes(target_dir)
            for code in grouped_codes:
                if code not in existing_now and code not in failed:
                    chunk_error = f"{code}: not_written"
                    last_error = chunk_error
                    failed[code] = "not_written"

            if include_existing:
                completed = len(processed_codes - set(failed))
                remaining = max(len(pending_codes) - len(processed_codes), 0)
            else:
                completed = len(existing_now & set(pending_codes))
                remaining = max(len(pending_codes) - completed, 0)
            _write_update_progress(
                progress_path,
                {
                    "status": "running",
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "target_symbols": len(codes),
                    "initial_existing": len(initial_existing & set(codes)),
                    "pending_symbols": len(pending_codes),
                    "chunk_size": chunk_size,
                    "sleep_seconds": sleep_seconds,
                    "completed_symbols": completed,
                    "failed_symbols": len(failed),
                    "remaining_symbols": remaining,
                    "last_chunk": batch_codes,
                    "current_symbol": grouped_codes[-1] if grouped_codes else "",
                    "current_start_date": code_start,
                    "last_error": chunk_error,
                },
            )
        if remaining == 0:
            break
        if sleep_seconds > 0 and (max_chunks is None or chunks_run < max_chunks):
            logger.info("Sleeping %.1f seconds before next chunk.", sleep_seconds)
            time.sleep(sleep_seconds)

    existing_final = _existing_stock_codes(target_dir)
    if include_existing:
        completed_final = len(processed_codes - set(failed))
        remaining_final = max(len(pending_codes) - len(processed_codes), 0)
    else:
        completed_final = len(existing_final & set(pending_codes))
        remaining_final = max(len(pending_codes) - completed_final, 0)
    status = "error" if failed else ("complete" if remaining_final == 0 else "partial")
    if failed and not last_error:
        first_code, first_reason = next(iter(failed.items()))
        last_error = f"{first_code}: {first_reason}"
    _write_update_progress(
        progress_path,
        {
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "target_symbols": len(codes),
            "initial_existing": len(initial_existing & set(codes)),
            "pending_symbols": len(pending_codes),
            "chunk_size": chunk_size,
            "sleep_seconds": sleep_seconds,
            "completed_symbols": completed_final,
            "failed_symbols": len(failed),
            "remaining_symbols": remaining_final,
            "last_chunk": [],
            "current_symbol": "",
            "last_error": last_error,
        },
    )
    return written


def _group_codes_by_start(pending: dict[str, tuple[Path, str, bool]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for code, (_path, actual_start, _needs_adj_backfill) in pending.items():
        grouped.setdefault(actual_start, []).append(code)
    return grouped


def _group_chunk_codes_by_start(chunk_starts: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for code, actual_start in chunk_starts.items():
        grouped.setdefault(actual_start, []).append(code)
    return grouped


def _limit_new_symbols_per_run(
    pending: dict[str, tuple[Path, str, bool]],
    max_new_symbols: int | None,
) -> dict[str, tuple[Path, str, bool]]:
    if max_new_symbols is None or max_new_symbols < 0:
        return pending
    existing = {code: item for code, item in pending.items() if item[0].exists()}
    new_items = {code: item for code, item in pending.items() if not item[0].exists()}
    if len(new_items) <= max_new_symbols:
        return pending
    selected_new = dict(list(new_items.items())[:max_new_symbols])
    skipped = len(new_items) - len(selected_new)
    logger.info("Deferring %d new stock files to later runs.", skipped)
    return {**existing, **selected_new}


def _batched(values: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def _existing_stock_codes(raw_dir: Path) -> set[str]:
    return {
        path.stem.upper()
        for path in raw_dir.glob("*.csv")
        if path.name.upper().endswith((".SZ.CSV", ".SH.CSV"))
    }


def _write_update_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_universe_list_dates(config: dict) -> dict[str, str]:
    data_cfg = config.get("data", {})
    path = resolve_path(data_cfg.get("constituents_file", "data/raw/mainboard_a_stocks.csv"))
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "list_date" not in df.columns:
        return {}
    code_col = _code_column(df)
    dates = pd.to_datetime(df["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    return {
        str(code).upper(): date.strftime("%Y-%m-%d")
        for code, date in zip(df[code_col], dates)
        if pd.notna(date)
    }


def _symbol_start_date(default_start: str, list_date: str | None) -> str:
    if not list_date:
        return default_start
    return max(pd.Timestamp(default_start), pd.Timestamp(list_date)).strftime("%Y-%m-%d")


def _date_windows(
    start_date: str | datetime,
    end_date: str | datetime,
    window_days: int | None,
) -> Iterable[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if window_days is None or window_days <= 0 or start > end:
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        return

    current = start
    step = pd.Timedelta(days=window_days - 1)
    while current <= end:
        window_end = min(current + step, end)
        yield current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        current = window_end + pd.Timedelta(days=1)


def _incremental_start(path: Path, configured_start: str) -> str:
    if not path.exists():
        return configured_start
    df = pd.read_csv(path, usecols=["trade_date"], parse_dates=["trade_date"])
    if df.empty:
        return configured_start
    return (df["trade_date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _needs_adj_factor_backfill(path: Path) -> bool:
    if not path.exists():
        return False
    columns = pd.read_csv(path, nrows=0).columns
    return "adj_factor" not in columns


def describe_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "<invalid-url>"
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or _default_port(parsed.scheme)}{path}"


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80
