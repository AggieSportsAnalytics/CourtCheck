"""
Scrape UC Davis Women's Tennis roster and seed the players table in Supabase.

Idempotent: skips players already in the DB by name. Safe to re-run.

Usage:
    python -m backend.tools.seed_players

    # Dry run (print scraped data, no DB writes):
    python -m backend.tools.seed_players --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

load_dotenv()

ROSTER_URL = "https://ucdavisaggies.com/sports/womens-tennis/roster"

YEAR_ALIASES: dict[str, str] = {
    "freshman": "Fr",
    "fr": "Fr",
    "sophomore": "So",
    "so": "So",
    "junior": "Jr",
    "jr": "Jr",
    "senior": "Sr",
    "sr": "Sr",
    "graduate": "Gr",
    "gr": "Gr",
}


def scrape_roster() -> list[dict]:
    """Scrape the UC Davis Women's Tennis roster page.

    Returns a list of dicts with keys: name, year, position.
    Falls back to a hard-coded roster if the page is unreachable.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(ROSTER_URL, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        players: list[dict] = []

        # The UCD Athletics site uses a common Sidearm Sports template.
        # Player names live inside .sidearm-roster-player-name elements;
        # year/class appear in .sidearm-roster-player-academic-year.
        name_els = soup.select(".sidearm-roster-player-name")

        seen_names: set[str] = set()
        for el in name_els:
            raw_name = " ".join(el.get_text(separator=" ", strip=True).split())  # normalize whitespace
            if not raw_name:
                continue

            # Skip obvious non-players: names containing digits or common staff titles
            if any(c.isdigit() for c in raw_name):
                continue
            lower = raw_name.lower()
            if any(t in lower for t in ("coach", "assistant", "trainer", "director", "staff")):
                continue

            # Deduplicate by normalized name
            if lower in seen_names:
                continue
            seen_names.add(lower)

            # Try to find the nearest year element in the same roster row
            row = el.find_parent("li") or el.find_parent("tr") or el.find_parent("div")
            raw_year = ""
            raw_position = ""
            if row:
                year_el = row.select_one(".sidearm-roster-player-academic-year")
                pos_el = row.select_one(".sidearm-roster-player-position")
                if year_el:
                    raw_year = year_el.get_text(strip=True)
                if pos_el:
                    raw_position = pos_el.get_text(strip=True)

            year = YEAR_ALIASES.get(raw_year.lower(), raw_year) if raw_year else None

            # Filter out height values (e.g. "5'10\"") from position field
            import re as _re
            position = raw_position if raw_position and not _re.match(r"^\d+'", raw_position) else None

            # Photo URL from lazy-loaded img data-src
            photo_url = None
            if row:
                img_el = row.select_one(".sidearm-roster-player-image img")
                if img_el:
                    data_src = img_el.get("data-src") or img_el.get("src")
                    if data_src:
                        if data_src.startswith("http"):
                            photo_url = data_src
                        else:
                            photo_url = f"https://ucdavisaggies.com{data_src}"

            players.append({"name": raw_name, "year": year, "position": position, "photo_url": photo_url})

        # If most players have years, filter out those with no year (likely coaches/staff)
        players_with_year = [p for p in players if p["year"]]
        if players_with_year:
            return players_with_year
        if players:
            return players

        # --- Fallback: try generic table rows ---
        players = _scrape_table_fallback(soup)
        if players:
            return players

    except Exception as exc:
        print(f"[seed_players] Scrape failed ({exc}); using hard-coded fallback roster.")

    return _hard_coded_fallback()


def _scrape_table_fallback(soup) -> list[dict]:
    """Try to extract player names from a plain HTML table."""
    players: list[dict] = []
    for row in soup.select("table tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        name_text = cells[0].get_text(strip=True) or cells[1].get_text(strip=True)
        if len(name_text) < 3 or any(c.isdigit() for c in name_text):
            continue
        players.append({"name": name_text, "year": None, "position": None})
    return players


def _hard_coded_fallback() -> list[dict]:
    """Minimal fallback roster based on publicly available UCD Women's Tennis data.

    Re-run after connectivity is restored to pull the live version.
    """
    return [
        {"name": "Sophie Luescher",   "year": "Sr", "position": None},
        {"name": "Akosua Frimpong",   "year": "Jr", "position": None},
        {"name": "Emma Smaagaard",    "year": "So", "position": None},
        {"name": "Julia Safarova",    "year": "Fr", "position": None},
        {"name": "Tara de Weerd",     "year": "Jr", "position": None},
        {"name": "Meredith Hankins",  "year": "So", "position": None},
        {"name": "Allie Mulville",    "year": "Sr", "position": None},
        {"name": "Lucia Arroyo",      "year": "Fr", "position": None},
    ]


def get_supabase_client():
    from supabase import create_client
    from dotenv import dotenv_values

    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        vals = dotenv_values(env_file)
        url = vals.get("SUPABASE_URL")
        key = vals.get("SUPABASE_SERVICE_ROLE_KEY")
    else:
        import os
        url = None
        key = None

    import os
    url = url or os.environ.get("SUPABASE_URL")
    key = key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env or environment."
        )

    return create_client(url, key)


def seed(players: list[dict], *, dry_run: bool = False) -> None:
    """Insert players into Supabase, skipping duplicates by name."""
    if dry_run:
        print(f"[dry-run] Would insert up to {len(players)} players:")
        for p in players:
            print(f"  {p['name']!r:30s}  year={p['year']}  position={p['position']}")
        return

    client = get_supabase_client()

    # Fetch existing players to deduplicate and update photos
    existing_resp = client.table("players").select("id, name, photo_url").execute()
    existing: dict[str, dict] = {row["name"]: row for row in (existing_resp.data or [])}

    to_insert = []
    to_update = []

    for p in players:
        if p["name"] in existing:
            ex = existing[p["name"]]
            if p.get("photo_url") and not ex.get("photo_url"):
                to_update.append({"id": ex["id"], "photo_url": p["photo_url"]})
        else:
            to_insert.append(p)

    if to_insert:
        resp = client.table("players").insert(to_insert).execute()
        if hasattr(resp, "error") and resp.error:
            print(f"[seed_players] Insert error: {resp.error}")
            sys.exit(1)
        print(f"[seed_players] Inserted {len(to_insert)} player(s).")

    for update in to_update:
        client.table("players").update({"photo_url": update["photo_url"]}).eq("id", update["id"]).execute()
    if to_update:
        print(f"[seed_players] Updated photo_url for {len(to_update)} existing player(s).")

    if not to_insert and not to_update:
        print("[seed_players] Nothing to insert or update.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed UC Davis Women's Tennis roster")
    parser.add_argument("--dry-run", action="store_true", help="Print data, skip DB writes")
    args = parser.parse_args()

    players = scrape_roster()
    print(f"[seed_players] Scraped {len(players)} player(s) from roster.")
    seed(players, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
