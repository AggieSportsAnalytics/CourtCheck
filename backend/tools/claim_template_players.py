"""Copy template players into Brian's roster and remap only his recordings.

Dry-run by default; --apply performs the printed inserts and updates. Never deletes.
Uses only Python's standard library and PostgREST.

From the repository root, load credentials before running:
  set -a; source frontend/web/.env.local; set +a
  python3 backend/tools/claim_template_players.py
  python3 backend/tools/claim_template_players.py --apply

Reads SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) and SUPABASE_SERVICE_ROLE_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter

BRIAN_USER_ID = "6e9abfcb-b30d-44ef-9789-0dccd95cacad"
COPY_FIELDS = ("name", "position", "year", "photo_url", "handedness")
PAGE_SIZE = 1000
REQUEST_TIMEOUT_SEC = 30


def _load_env() -> tuple[str, str]:
    url = (os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Set SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL to your HTTPS Supabase URL.")
    if not key:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required; see the environment setup above.")
    return url, key


class Client:
    def __init__(self, url: str, key: str) -> None:
        self.url = url
        self.headers = {
            "apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "Prefer": "return=representation",
        }

    def call(self, method: str, table: str, params: dict, body=None) -> list[dict]:
        path = f"/rest/v1/{table}?{urllib.parse.urlencode(params)}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
                rows = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{method} {table} failed (HTTP {error.code}): {error.read().decode()[:300]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"{method} {table} could not reach Supabase: {error.reason}") from error
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Unexpected {table} response; stopped without further changes.")
        return rows

    def auth_user(self, user_id: str) -> dict:
        request = urllib.request.Request(f"{self.url}/auth/v1/admin/users/{user_id}", headers=self.headers, method="GET")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
            user = json.load(response)
        if not isinstance(user, dict) or user.get("id") != user_id:
            raise ValueError("Auth user lookup did not return the expected account; stopped.")
        return user

    def set_user_metadata(self, user_id: str, metadata: dict) -> dict:
        data = json.dumps({"user_metadata": metadata}).encode()
        request = urllib.request.Request(f"{self.url}/auth/v1/admin/users/{user_id}", data=data, headers=self.headers, method="PUT")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
            return json.load(response)

    def rows(self, table: str, params: dict) -> list[dict]:
        result = []
        while True:
            page = self.call("GET", table, {**params, "order": "id.asc", "limit": PAGE_SIZE, "offset": len(result)})
            result.extend(page)
            if len(page) < PAGE_SIZE:
                return result


def _check_players(rows: list[dict], owner: str | None) -> None:
    for row in rows:
        uuid.UUID(row["id"])
        if "user_id" not in row or row["user_id"] != owner or not isinstance(row.get("name"), str):
            raise ValueError("Player ownership or name could not be established; stopped.")


def _plan(client: Client) -> list[dict]:
    columns = ",".join(("id", "user_id", *COPY_FIELDS))
    templates = client.rows("players", {"select": columns, "user_id": "is.null"})
    owned = client.rows("players", {"select": columns, "user_id": f"eq.{BRIAN_USER_ID}"})
    _check_players(templates, None)
    _check_players(owned, BRIAN_USER_ID)
    matches = client.rows("matches", {"select": "id,user_id,player_id", "user_id": f"eq.{BRIAN_USER_ID}"})
    if any(row.get("user_id") != BRIAN_USER_ID for row in matches):
        raise ValueError("Recording ownership could not be established; stopped.")
    counts = Counter(row.get("player_id") for row in matches)
    by_name = {}
    for player in owned:
        by_name.setdefault(player["name"], player["id"])
    actions = []
    for template in templates:
        name = template["name"]
        create = name not in by_name
        target = by_name.get(name) or str(uuid.uuid4())
        by_name[name] = target
        actions.append({"template": template, "target": target, "create": create, "matches": counts[template["id"]]})
    return actions


def _print(actions: list[dict], applied: bool = False) -> None:
    print("Action  Player                         Template ID                          Owned ID                             Recordings")
    for action in actions:
        verb = ("CREATED" if applied else "CREATE") if action["create"] else "REUSE"
        print(f"{verb:7} {action['template']['name']:30} {action['template']['id']} {action['target']} {action['matches']:10}")
    print(f"Templates: {len(actions)}; copies: {sum(a['create'] for a in actions)}; "
          f"reused: {sum(not a['create'] for a in actions)}; recordings remapped: {sum(a['matches'] for a in actions)}")
    print("ID mapping:")
    for action in actions:
        print(f"  {action['template']['id']} -> {action['target']}")


def _apply(client: Client, actions: list[dict]) -> list[dict]:
    applied = []
    resolved = {}
    for action in actions:
        template = action["template"]
        target = resolved.get(template["name"], action["target"])
        created = False
        if action["create"]:
            # Recheck names so retries after a partial run reuse existing copies.
            existing = client.rows("players", {"select": "id,name,user_id", "user_id": f"eq.{BRIAN_USER_ID}"})
            _check_players(existing, BRIAN_USER_ID)
            same_name = next((row for row in existing if row["name"] == template["name"]), None)
            if same_name:
                target = same_name["id"]
            else:
                row = {field: template.get(field) for field in COPY_FIELDS}
                inserted = client.call("POST", "players", {}, {**row, "id": target, "user_id": BRIAN_USER_ID})
                _check_players(inserted, BRIAN_USER_ID)
                if len(inserted) != 1 or inserted[0]["id"] != target:
                    raise ValueError("The owned player insert was not confirmed; stopped.")
                created = True
        resolved[template["name"]] = target
        updated = client.call("PATCH", "matches", {
            "user_id": f"eq.{BRIAN_USER_ID}", "player_id": f"eq.{template['id']}",
        }, {"player_id": target})
        if any(row.get("user_id") != BRIAN_USER_ID or row.get("player_id") != target for row in updated):
            raise ValueError("The recording remap was not confirmed; stopped.")
        result = {**action, "target": target, "create": created, "matches": len(updated)}
        applied.append(result)
        print(f"Applied: {template['id']} -> {target}; {len(updated)} recordings")
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="execute the plan (default: dry run)")
    args = parser.parse_args()
    try:
        client = Client(*_load_env())
        actions = _plan(client)
        print(f"User: {BRIAN_USER_ID}; {'apply' if args.apply else 'dry run'}")
        _print(actions)
        current = (client.auth_user(BRIAN_USER_ID).get("user_metadata") or {})
        template_flag = current.get("onboarding_template")
        print(f"user_metadata.onboarding_template: {template_flag!r} -> 'uc-davis' "
              "(hides the shared template rows once the owned copies exist)")
        if args.apply:
            _print(_apply(client, actions), applied=True)
            if template_flag != "uc-davis":
                updated = client.set_user_metadata(BRIAN_USER_ID, {**current, "onboarded": True, "onboarding_template": "uc-davis"})
                print(f"Applied: user_metadata.onboarding_template = {(updated.get('user_metadata') or {}).get('onboarding_template')!r}")
        else:
            print("Dry run only. IDs for new copies are proposed; re-run with --apply to execute.")
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        sys.exit(f"Stopped: {error}. Review any printed applied actions before retrying.")


if __name__ == "__main__":
    main()
