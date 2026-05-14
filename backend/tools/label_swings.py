"""
Swing clip labeling UI — Streamlit app backed by Supabase.

Labels are written directly to the swing_labels DB table — no local CSV needed.
Multiple annotators can run simultaneously; each write is an independent DB update.

Usage (local):
    streamlit run backend/tools/label_swings.py -- --annotator brian

Usage (Streamlit Community Cloud):
    Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the app secrets.
    No other arguments needed.

Optional flags:
    --annotator <name>   Your name (default: "unknown")
    --filter all         Show already-labeled clips too (default: unlabeled only)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import (
    SWING_CLIPS_BUCKET,
    db_get_all,
    db_save_label,
    get_supabase_client,
)

LABELS = ["forehand", "backhand", "serve", "volley", "unclear", "skip"]
LABEL_ICONS = {
    "forehand": "🟢",
    "backhand": "🔵",
    "serve": "🔴",
    "volley": "🟠",
    "unclear": "⚪",
    "skip": "⏭️",
}


# ── Supabase client — cached for the lifetime of the Streamlit server process ──

@st.cache_resource
def _supabase():
    return get_supabase_client()


# ── Signed URL generation ──────────────────────────────────────────────────────

def _signed_url(supabase_path: str) -> str | None:
    if not supabase_path:
        return None
    result = _supabase().storage.from_(SWING_CLIPS_BUCKET).create_signed_url(
        supabase_path, expires_in=3600
    )
    return result.get("signedURL") or result.get("signedUrl")


# ── Argument parsing (Streamlit passes extra args after --) ────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", default="unknown")
    parser.add_argument("--filter", default="unlabeled", choices=["unlabeled", "all"])
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="CourtCheck Annotation", layout="wide")
    args = _parse_args()

    # Annotator name — CLI arg takes precedence, otherwise prompt in sidebar
    with st.sidebar:
        st.title("CourtCheck")
        if args.annotator != "unknown":
            annotator = args.annotator
            st.caption(f"Annotator: **{annotator}**")
        else:
            annotator = st.text_input("Your name", placeholder="e.g. kenny").strip().lower()
            if not annotator:
                st.warning("Enter your name to start.")
                st.stop()

    # Fetch all rows every rerun so sidebar stats stay current across annotators.
    with st.spinner("Loading queue…"):
        df = db_get_all(client=_supabase())

    if df.empty:
        st.error("No clips in the database. Run seed_swing_labels.py first.")
        return

    # Each annotator sees their own assigned clips plus the shared pool
    # (assigned_to == ""). Pool clips are first-come-first-serve once labeled.
    has_assignments = "assigned_to" in df.columns and (df["assigned_to"] != "").any()
    if has_assignments:
        mine = df["assigned_to"] == annotator
        pool = df["assigned_to"] == ""
        # Pool rows already labeled by someone else are dropped so two people
        # can't fight over the same clip.
        pool_taken = pool & (df["label"] != "")
        my_pool = df[mine | (pool & ~pool_taken)].reset_index(drop=True)
    else:
        my_pool = df

    queue = (
        my_pool[my_pool["label"] == ""].reset_index(drop=True)
        if args.filter == "unlabeled"
        else my_pool.reset_index(drop=True)
    )

    # Sidebar — progress and class counts (scoped to this annotator)
    with st.sidebar:
        st.divider()
        st.write("**Your progress**")
        my_total = len(my_pool)
        my_labeled = my_pool["label"].isin(
            ["forehand", "backhand", "serve", "volley", "unclear"]
        ).sum()
        st.metric("Labeled", f"{my_labeled} / {my_total}")
        st.progress(int(my_labeled) / my_total if my_total else 0)
        st.divider()
        st.write("**Your label counts**")
        for label in ["forehand", "backhand", "serve", "volley"]:
            count = (my_pool["label"] == label).sum()
            st.write(f"{LABEL_ICONS[label]} {label.capitalize()}: {count}")
        st.divider()
        if has_assignments and my_total == 0:
            st.error(f"No clips assigned to **{annotator}**. Check your name.")
            st.stop()
        st.caption(f"Annotator: **{annotator}**")

    if queue.empty:
        st.success(f"All {my_total} clips labeled!")
        return

    # Session state tracks position in the queue
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if st.session_state.idx >= len(queue):
        st.session_state.idx = 0

    row = queue.iloc[st.session_state.idx]

    st.title(f"Clip {st.session_state.idx + 1} / {len(queue)} remaining")
    st.caption(
        f"Source: `{row['source_video']}`  |  "
        f"Frame: {row['peak_frame']}  |  "
        f"Player: {row['player_idx']}  |  "
        f"Wrist velocity: {row['wrist_velocity']} px/fr"
    )

    # Video player — HTML embed is more reliable than st.video() with external URLs
    try:
        url = _signed_url(str(row.get("supabase_path", "")).strip())
        if url is None:
            st.warning(f"No supabase_path for clip `{row['clip_id']}` — skipping")
        else:
            st.components.v1.html(
                f"""
                <video
                    src="{url}"
                    autoplay loop muted playsinline
                    style="width:100%; max-height:480px; background:#000;"
                    controls
                ></video>
                """,
                height=500,
            )
    except Exception as e:
        st.error(f"Could not load clip `{row['clip_id']}`: {e}")

    # Label buttons
    st.divider()
    cols = st.columns(len(LABELS))
    for col, label in zip(cols, LABELS):
        icon = LABEL_ICONS[label]
        if col.button(f"{icon} {label.capitalize()}", use_container_width=True, key=label):
            db_save_label(row["clip_id"], label, annotator, client=_supabase())
            st.session_state.idx += 1
            st.rerun()

    # Navigation
    nav_cols = st.columns([1, 8])
    if nav_cols[0].button("← Back", disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.rerun()


if __name__ == "__main__":
    main()
