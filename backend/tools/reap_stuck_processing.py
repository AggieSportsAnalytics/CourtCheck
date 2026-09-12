"""Reap stuck 'processing' recordings — matches Modal started but never finished
(container timeout at 1800s, OOM, or spot preemption), leaving the row wedged at
status='processing' forever while the UI polls it indefinitely.

A hard container kill (SIGKILL on timeout) never runs run_pipeline's except
block, so the pipeline cannot self-heal these; this external sweep is the
backstop.

Behaviour:
  - Finds rows where status='processing' AND created_at is older than the
    age window (default 45 min).
  - Flips them to status='failed' with a retryable error so the coach can
    re-run. The write is guarded on status='processing' so a run that finished
    between our read and write is never clobbered.

Safety:
  - Dry-run by default. `--apply` required to actually write.
  - The age window MUST exceed Modal's 30-min pipeline timeout so a live run is
    never reaped mid-flight. created_at is used because matches has no
    updated_at column yet (deferred schema work).
  - Prints every action.

Usage:
    python -m backend.tools.reap_stuck_processing                    # dry run
    python -m backend.tools.reap_stuck_processing --apply
    python -m backend.tools.reap_stuck_processing --max-age-minutes 60 --apply
    python -m backend.tools.reap_stuck_processing --match-id <uuid> --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from supabase import create_client


# Must exceed Modal's 1800s (30 min) pipeline timeout so an in-flight run is
# never mistaken for a dead one.
DEFAULT_MAX_AGE_MINUTES = 45

FAILURE_MESSAGE = "Processing timed out. Please try again."


def _get_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
        sys.exit(1)
    return create_client(url, key)


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    """Parse a Supabase timestamp into an aware UTC datetime, or None if unusable."""
    if not ts:
        return None
    s = ts.strip()
    # Python < 3.11 fromisoformat rejects a trailing 'Z'; normalise it.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_stuck(created_at: Optional[str], now: datetime, max_age_minutes: int) -> bool:
    """True if a processing row's created_at is older than the age window."""
    created = _parse_ts(created_at)
    if created is None:
        return False
    return (now - created) > timedelta(minutes=max_age_minutes)


def reap(match_ids: Optional[Iterable[str]], apply: bool, max_age_minutes: int) -> None:
    supabase = _get_supabase()
    now = datetime.now(timezone.utc)

    query = (
        supabase.table("matches")
        .select("id, name, status, created_at, progress")
        .eq("status", "processing")
    )
    if match_ids:
        query = query.in_("id", list(match_ids))
    rows = query.execute().data or []

    # When the operator names specific ids, target them regardless of age.
    if match_ids:
        stuck = rows
    else:
        stuck = [r for r in rows if is_stuck(r.get("created_at"), now, max_age_minutes)]

    if not stuck:
        print(
            f"No stuck 'processing' rows found "
            f"(age window: {max_age_minutes} min). Nothing to reap."
        )
        return

    print(f"Found {len(stuck)} stuck row(s). Mode: {'APPLY' if apply else 'DRY RUN'}\n")

    reaped = 0
    failed = 0
    for row in stuck:
        mid = row["id"]
        name = row.get("name") or "(unnamed)"
        created = row.get("created_at")
        print(f"- {mid} {name!r} (created {created}, progress {row.get('progress')})")

        if not apply:
            print("    would mark failed (dry run)")
            reaped += 1
            continue

        try:
            # Guard on status so a run that just finished is never clobbered.
            supabase.table("matches").update(
                {"status": "failed", "error": FAILURE_MESSAGE, "progress": 0}
            ).eq("id", mid).eq("status", "processing").execute()
            print("    marked failed")
            reaped += 1
        except Exception as e:
            print(f"    ! failed to update: {e}")
            failed += 1

    print()
    verb = "Reaped" if apply else "Would reap"
    print(f"{verb} {reaped} row(s) · failed {failed}")
    if not apply:
        print("Re-run with --apply to actually mark them failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually mark rows failed. Without this flag the script does a dry run.",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=int,
        default=DEFAULT_MAX_AGE_MINUTES,
        help=(
            f"Minimum age (by created_at) before a processing row is reaped. "
            f"Default {DEFAULT_MAX_AGE_MINUTES}. Must exceed Modal's 30-min timeout."
        ),
    )
    parser.add_argument(
        "--match-id",
        action="append",
        dest="match_ids",
        help="Restrict to one or more match ids. Repeatable. Ignores --max-age-minutes.",
    )
    args = parser.parse_args()
    reap(match_ids=args.match_ids, apply=args.apply, max_age_minutes=args.max_age_minutes)


if __name__ == "__main__":
    main()
