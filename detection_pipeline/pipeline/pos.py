"""POS correlation for BILLING_QUEUE_ABANDON.

Loads the Brigade Bangalore POS CSV and provides a `had_transaction(window)`
helper. We can't deterministically link a visitor to a POS row without a face
or member ID, so we use a temporal heuristic: if at least one POS transaction
was registered within `window_seconds` after the visitor LEFT the billing zone,
we do NOT flag abandonment. Otherwise we do.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd


class POSCorrelator:
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path, low_memory=False)
        if "invoice_type" in df.columns:
            df = df[df["invoice_type"].astype(str).str.lower() == "sales"].copy()
        ts = pd.to_datetime(
            df["order_date"].astype(str) + " " + df["order_time"].astype(str),
            format="%d-%m-%Y %H:%M:%S", errors="coerce", utc=True,
        )
        df["ts"] = ts
        df = df.dropna(subset=["ts"]).sort_values("ts")
        dedup_col = "invoice_number" if "invoice_number" in df.columns else "order_id"
        self.txn_times = (
            df.drop_duplicates(subset=[dedup_col])["ts"]
              .dt.tz_convert("UTC")
              .to_list()
        )

    def had_transaction(self, exit_time_utc: datetime,
                        window_seconds: int = 120) -> bool:
        end = exit_time_utc + timedelta(seconds=window_seconds)
        for t in self.txn_times:
            if exit_time_utc <= t <= end:
                return True
            if t > end:
                break
        return False
