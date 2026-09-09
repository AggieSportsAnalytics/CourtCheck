"""Reclaim Supabase storage and prune dead match rows.

Dry-run by default: prints exactly what would change and exits. Pass --apply to execute.

Tiers (each opt-in via a flag so the safe set can run alone):
  --safe        orphan result files (no match row), pending rows that never received an
                upload, the stale `processing` row (marked failed), raw videos of failed rows
  --raw-older N delete raw-videos for `done` matches older than N days (raw is only needed
                to reprocess; the rendered results stay)
  --match ID..  delete specific matches entirely (row + raw + results)

Usage:
  python -m backend.tools.cleanup_storage --safe                 # dry run
  python -m backend.tools.cleanup_storage --safe --apply
  python -m backend.tools.cleanup_storage --raw-older 30 --apply
  python -m backend.tools.cleanup_storage --match 482cfee0 b8f4aba2 --apply

Reads SUPABASE_URL / NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from the
environment (or frontend/web/.env.local via python-dotenv if installed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BUCKET_RAW = "raw-videos"
BUCKET_RESULTS = "results"
STALE_PROCESSING_DAYS = 2
PAGE = 1000


def _load_env() -> tuple[str, str]:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(Path(__file__).resolve().parents[2] / "frontend" / "web" / ".env.local")
    except ImportError:
        pass
    url = (os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required (see frontend/web/.env.local)")
    return url, key


class Client:
    def __init__(self, url: str, key: str) -> None:
        self.url = url
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def call(self, method: str, path: str, body=None, extra=None):
        headers = {**self.headers, **(extra or {})}
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"{method} {path} -> {err.code}: {err.read().decode()[:300]}") from err

    def list_objects(self, bucket: str, prefix: str = "") -> list[tuple[str, int, str]]:
        out: list[tuple[str, int, str]] = []
        offset = 0
        while True:
            rows = self.call(
                "POST",
                f"/storage/v1/object/list/{bucket}",
                {"prefix": prefix, "limit": PAGE, "offset": offset, "sortBy": {"column": "name", "order": "asc"}},
            )
            for obj in rows:
                name = f"{prefix}/{obj['name']}" if prefix else obj["name"]
                if obj.get("id") is None:
                    out.extend(self.list_objects(bucket, name))
                else:
                    out.append((name, (obj.get("metadata") or {}).get("size") or 0, (obj.get("created_at") or "")[:10]))
            if len(rows) < PAGE:
                return out
            offset += PAGE

    def delete_objects(self, bucket: str, names: list[str]) -> None:
        for i in range(0, len(names), 100):
            self.call("DELETE", f"/storage/v1/object/{bucket}", {"prefixes": names[i : i + 100]})


def _mb(n: int) -> str:
    return f"{n / 1e6:7.1f} MB"


def _plan(client: Client, args) -> list[dict]:
    matches = client.call("GET", "/rest/v1/matches?select=id,status,created_at,name,input_path&limit=1000")
    by_id = {m["id"]: m for m in matches}
    raw = client.list_objects(BUCKET_RAW)
    results = client.list_objects(BUCKET_RESULTS)
    have_storage = {n.split("/")[0] for n, _, _ in raw + results}
    actions: list[dict] = []

    if args.safe:
        orphan = [(n, s) for n, s, _ in results if n.split("/")[0] not in by_id]
        if orphan:
            actions.append({"kind": "delete_objects", "bucket": BUCKET_RESULTS, "names": [n for n, _ in orphan],
                            "bytes": sum(s for _, s in orphan), "why": "result files whose match row no longer exists"})
        cutoff = (datetime.now(timezone.utc) - timedelta(days=STALE_PROCESSING_DAYS)).isoformat()
        stale = [m["id"] for m in matches if m["status"] == "processing" and m["created_at"] < cutoff]
        for mid in stale:
            actions.append({"kind": "mark_failed", "id": mid,
                            "why": f"stuck in processing since {by_id[mid]['created_at'][:10]}"})
        failed = {m["id"] for m in matches if m["status"] == "failed"} | set(stale)
        raw_failed = [(n, s) for n, s, _ in raw if n.split("/")[0] in failed]
        if raw_failed:
            actions.append({"kind": "delete_objects", "bucket": BUCKET_RAW, "names": [n for n, _ in raw_failed],
                            "bytes": sum(s for _, s in raw_failed), "why": "raw video of failed / stale rows"})
        pending = [m["id"] for m in matches if m["status"] == "pending" and m["id"] not in have_storage]
        if pending:
            actions.append({"kind": "delete_rows", "ids": pending, "why": "pending rows that never received an upload"})

    if args.raw_older is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.raw_older)).isoformat()
        old_done = {m["id"] for m in matches if m["status"] == "done" and m["created_at"] < cutoff}
        raw_old = [(n, s) for n, s, _ in raw if n.split("/")[0] in old_done]
        if raw_old:
            actions.append({"kind": "delete_objects", "bucket": BUCKET_RAW, "names": [n for n, _ in raw_old],
                            "bytes": sum(s for _, s in raw_old),
                            "why": f"raw video of done matches older than {args.raw_older} days (results kept; reprocess needs re-upload)"})

    for prefix in args.match or []:
        full = [mid for mid in by_id if mid.startswith(prefix)]
        if len(full) != 1:
            sys.exit(f"--match {prefix}: expected exactly one match id, found {len(full)}")
        mid = full[0]
        for bucket, objs in ((BUCKET_RAW, raw), (BUCKET_RESULTS, results)):
            mine = [(n, s) for n, s, _ in objs if n.split("/")[0] == mid]
            if mine:
                actions.append({"kind": "delete_objects", "bucket": bucket, "names": [n for n, _ in mine],
                                "bytes": sum(s for _, s in mine), "why": f"--match {prefix} ({by_id[mid].get('name') or by_id[mid].get('input_path')})"})
        actions.append({"kind": "delete_rows", "ids": [mid], "why": f"--match {prefix}"})
    return actions


def _print(actions: list[dict]) -> None:
    total = 0
    for a in actions:
        if a["kind"] == "delete_objects":
            total += a["bytes"]
            print(f"DELETE {len(a['names']):4d} objects from {a['bucket']:11s} {_mb(a['bytes'])}  {a['why']}")
            for n in a["names"][:5]:
                print(f"         {n}")
            if len(a["names"]) > 5:
                print(f"         ... {len(a['names']) - 5} more")
        elif a["kind"] == "delete_rows":
            print(f"DELETE {len(a['ids']):4d} match rows                       {a['why']}")
        elif a["kind"] == "mark_failed":
            print(f"UPDATE match {a['id'][:8]} -> failed                     {a['why']}")
    print(f"\nReclaims {_mb(total)} of storage.")


def _apply(client: Client, actions: list[dict]) -> None:
    for a in actions:
        if a["kind"] == "delete_objects":
            client.delete_objects(a["bucket"], a["names"])
        elif a["kind"] == "delete_rows":
            client.call("DELETE", "/rest/v1/matches?id=in.(" + ",".join(a["ids"]) + ")", None, {"Prefer": "return=minimal"})
        elif a["kind"] == "mark_failed":
            client.call("PATCH", f"/rest/v1/matches?id=eq.{a['id']}",
                        {"status": "failed", "error": "Processing never completed. Upload the video again to reprocess."},
                        {"Prefer": "return=minimal"})
        print("applied:", a["kind"], a.get("why"))


def _summary(client: Client) -> None:
    matches = client.call("GET", "/rest/v1/matches?select=status&limit=1000")
    print("matches:", dict(Counter(m["status"] for m in matches)))
    for bucket in (BUCKET_RAW, BUCKET_RESULTS):
        objs = client.list_objects(bucket)
        print(f"{bucket:11s} {len(objs):4d} objects {_mb(sum(s for _, s, _ in objs))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--safe", action="store_true")
    parser.add_argument("--raw-older", type=int, metavar="DAYS")
    parser.add_argument("--match", nargs="*", metavar="ID_PREFIX")
    parser.add_argument("--apply", action="store_true", help="execute (default is dry run)")
    args = parser.parse_args()
    if not (args.safe or args.raw_older is not None or args.match):
        parser.error("choose at least one of --safe, --raw-older, --match")
    client = Client(*_load_env())
    print("== before"); _summary(client)
    actions = _plan(client, args)
    print("\n== plan" if not args.apply else "\n== applying")
    _print(actions)
    if not args.apply:
        print("\nDry run. Re-run with --apply to execute.")
        return
    _apply(client, actions)
    print("\n== after"); _summary(client)


if __name__ == "__main__":
    main()
