from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.factor_calculator import load_or_compute_factors
from src.scoring import build_strategy_scores
from src.strategy import select_stocks


def read_previous_holdings(path: str | Path | None = None) -> list[str]:
    config = load_config()
    holdings_path = resolve_path(path or config["outputs"]["holdings_file"])
    if not holdings_path.exists():
        return []
    df = pd.read_csv(holdings_path)
    col = "instrument" if "instrument" in df.columns else "ticker"
    if col not in df.columns:
        return []
    return df[col].dropna().astype(str).tolist()


def generate_signal(
    signal_date: str,
    previous_holdings: list[str] | None = None,
    factor_file: str | Path | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = config or load_config()
    data_cfg = config["data"]
    strategy_cfg = config["strategy"]
    use_latest_date = str(signal_date).lower() == "latest"
    required_end_date = _required_latest_data_date(data_cfg) if use_latest_date else None
    factor_end_date = required_end_date.strftime("%Y-%m-%d") if required_end_date is not None else signal_date

    factors = load_or_compute_factors(
        start_date=data_cfg["start_date"],
        end_date=factor_end_date,
        cache_file=factor_file or config["factors"]["cache_file"],
    )
    scores = build_strategy_scores(factors, config)
    latest_factor_date = _latest_index_date(factors, "factor")
    latest_score_date = _latest_index_date(scores, "score")
    latest_date = latest_score_date
    if use_latest_date:
        assert required_end_date is not None
        if latest_factor_date < required_end_date or latest_score_date < required_end_date:
            raise ValueError(
                "Latest factor/score data is stale: "
                f"latest factor date {latest_factor_date.date()}, "
                f"latest score date {latest_score_date.date()}, "
                f"required data date {required_end_date.date()}. "
                "Please update/convert price data and recompute factors before generating a latest signal."
            )
        signal_date = latest_date.strftime("%Y-%m-%d")
    else:
        requested_date = pd.Timestamp(signal_date).normalize()
        if latest_date != requested_date:
            raise ValueError(f"Factor cache latest date {latest_date.date()} does not match signal_date {requested_date.date()}.")
    latest_scores = scores.xs(latest_date, level=0, drop_level=True)
    previous_holdings = previous_holdings if previous_holdings is not None else read_previous_holdings()
    holdings = select_stocks(
        latest_scores,
        top_n=int(strategy_cfg.get("top_n", 7)),
        previous_holdings=previous_holdings or None,
        max_turnover=int(strategy_cfg.get("max_turnover", 1)),
        rank_buffer=int(strategy_cfg.get("rank_buffer", 0)),
    )

    old_set = set(previous_holdings or [])
    new_set = set(holdings)
    rows = []
    for code in holdings:
        rows.append({"date": signal_date, "instrument": code, "action": "HOLD" if code in old_set else "BUY"})
    for code in sorted(old_set - new_set):
        rows.append({"date": signal_date, "instrument": code, "action": "SELL"})
    return pd.DataFrame(rows), holdings


def _latest_index_date(frame: pd.DataFrame | pd.Series, label: str) -> pd.Timestamp:
    if frame.empty:
        raise ValueError(f"{label.capitalize()} data is empty.")
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError(f"Expected {label} data to use a MultiIndex of datetime/instrument.")
    return pd.Timestamp(frame.index.get_level_values(0).max()).normalize()


def _required_latest_data_date(data_cfg: dict, now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the minimum acceptable data date for a latest signal.

    A-share daily data is typically available after the latest trading day's
    close in the evening. Before the configured cutoff hour, require data up
    to the previous trading day; at/after the cutoff, require the latest
    trading day. This uses pandas business days as a lightweight trading-day
    approximation; holidays can be handled later by wiring a market calendar.
    """
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


def save_signal(signal_df: pd.DataFrame, holdings: list[str], signal_date: str, config: dict | None = None) -> tuple[Path, Path]:
    config = config or load_config()
    out_dir = resolve_path(config["outputs"].get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"signal_{signal_date}.csv"
    holdings_path = resolve_path(config["outputs"]["holdings_file"])
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"instrument": holdings}).to_csv(holdings_path, index=False, encoding="utf-8-sig")
    return signal_path, holdings_path
