"""Shared I/O helpers for swing annotation tools.

CSV helpers  — used by extract_swings_local.py and upload_clips.py.
DB helpers   — used by label_swings.py and seed_swing_labels.py.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "clip_id",
    "supabase_path",
    "source_video",
    "peak_frame",
    "window_start",
    "window_end",
    "player_idx",
    "wrist_velocity",
    "label",
    "annotator",
    "labeled_at",
]
SWING_CLIPS_BUCKET = "swing-clips"
SWING_LABELS_TABLE = "swing_labels"

# ── CSV helpers (local tools only) ───────────────────────────────────────────

def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(path, dtype=str).fillna("")


def write_manifest(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append new manifest rows, writing header if file is new."""
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Supabase DB helpers ───────────────────────────────────────────────────────

def get_supabase_client():
    """Return a Supabase client.

    Reads credentials directly from the .env file to avoid stale shell env vars
    overriding load_dotenv() — same pattern as upload_clips.py.
    Falls back to os.environ (needed when running on Streamlit Community Cloud).
    """
    from supabase import create_client
    from dotenv import dotenv_values

    # Walk up to find the project root .env (two levels above this file: tools/ → backend/ → project root)
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        vals = dotenv_values(env_file)
        url = vals.get("SUPABASE_URL") or os.environ["SUPABASE_URL"]
        key = vals.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    else:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    return create_client(url, key)


def db_get_all(client=None) -> pd.DataFrame:
    """Fetch every row from swing_labels. Handles Supabase 1 000-row page limit."""
    client = client or get_supabase_client()
    PAGE = 1000
    rows: list[dict] = []
    start = 0
    while True:
        result = (
            client.table(SWING_LABELS_TABLE)
            .select("*")
            .range(start, start + PAGE - 1)
            .execute()
        )
        rows.extend(result.data)
        if len(result.data) < PAGE:
            break
        start += PAGE

    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    df = pd.DataFrame(rows)
    # Ensure all expected columns exist and fill nulls
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[MANIFEST_COLUMNS].fillna("")
    return df


def db_save_label(clip_id: str, label: str, annotator: str, client=None) -> None:
    """Write a label for one clip to the DB."""
    client = client or get_supabase_client()
    client.table(SWING_LABELS_TABLE).update({
        "label": label,
        "annotator": annotator,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }).eq("clip_id", clip_id).execute()


def db_seed_from_csv(csv_path: Path, client=None) -> int:
    """Bulk-upsert all rows from a manifest CSV into swing_labels. Returns row count."""
    import json

    client = client or get_supabase_client()
    df = read_manifest(csv_path)

    # Cast numeric columns so Supabase accepts them as numbers, not strings
    for col in ("peak_frame", "window_start", "window_end", "player_idx"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["wrist_velocity"] = (
        pd.to_numeric(df["wrist_velocity"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1e9)  # guard against inf
    )

    # Replace empty labeled_at with None (DB expects NULL, not empty string)
    df["labeled_at"] = df["labeled_at"].replace("", None)

    # Serialize via pandas JSON to convert numpy types → Python native types
    records: list[dict] = json.loads(df.to_json(orient="records"))

    CHUNK = 500
    inserted = 0
    for i in range(0, len(records), CHUNK):
        chunk = records[i : i + CHUNK]
        client.table(SWING_LABELS_TABLE).upsert(chunk, on_conflict="clip_id").execute()
        inserted += len(chunk)
    return inserted
