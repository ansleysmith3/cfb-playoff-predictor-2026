"""Pull raw college football data from the CollegeFootballData (CFBD) API.

Usage:
    python -m src.ingest            # defaults to the 2026 season
    python -m src.ingest --year 2025
    python -m src.ingest --years 2014-2026
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.collegefootballdata.com"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_api_key() -> str:
    """Read the key from .env locally or from the environment (GitHub Actions secret)."""
    # override=True so a stale shell variable can't shadow the .env file locally.
    # In GitHub Actions there is no .env, so the secret from the environment is used.
    load_dotenv(override=True)
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def fetch(endpoint: str, params: dict, key: str, retries: int = 4) -> list[dict]:
    """GET one CFBD endpoint and return the parsed JSON.

    Retries with exponential backoff on 429/5xx, which are usually temporary.
    """
    for attempt in range(retries + 1):
        resp = requests.get(
            f"{BASE_URL}{endpoint}",
            params=params,
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            wait = 2 ** (attempt + 1)
            print(f"{endpoint} returned {resp.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()


def ingest_games(year: int, key: str) -> pd.DataFrame:
    frames = []
    for season_type in ("regular", "postseason"):
        data = fetch("/games", {"year": year, "seasonType": season_type}, key)
        frames.append(pd.json_normalize(data))
    return pd.concat(frames, ignore_index=True)


def ingest_rankings(year: int, key: str) -> pd.DataFrame:
    """Flatten the nested poll structure into one row per (week, poll, team)."""
    rows = []
    for week in fetch("/rankings", {"year": year}, key):
        for poll in week.get("polls", []):
            for r in poll.get("ranks", []):
                rows.append(
                    {
                        "season": week.get("season"),
                        "seasonType": week.get("seasonType"),
                        "week": week.get("week"),
                        "poll": poll.get("poll"),
                        "rank": r.get("rank"),
                        "school": r.get("school"),
                        "conference": r.get("conference"),
                        "points": r.get("points"),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--years", help="Inclusive range such as 2014-2025; overrides --year"
    )
    args = parser.parse_args()

    if args.years:
        first, last = (int(y) for y in args.years.split("-"))
        years = list(range(first, last + 1))
    else:
        years = [args.year]

    key = get_api_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"--- {year} ---")
        games = ingest_games(year, key)
        games.to_csv(RAW_DIR / f"games_{year}.csv", index=False)
        print(f"Saved {len(games)} games")

        rankings = ingest_rankings(year, key)
        rankings.to_csv(RAW_DIR / f"rankings_{year}.csv", index=False)
        print(f"Saved {len(rankings)} ranking rows")


if __name__ == "__main__":
    main()
