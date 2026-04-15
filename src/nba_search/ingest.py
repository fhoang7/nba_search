import argparse
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any
import json 
import logging

import requests
from requests.exceptions import RequestException
from nba_search.models import GameDocument, Performer

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
LOOKBACK_DAYS = 3
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "corpus.jsonl"
REQUEST_TIMEOUT = 10


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["game_id"] for line in path.open()}

def fetch_game_ids_for_date(date_str: str) -> list[str]:
    """Return event IDs for all Final games on a given YYYYMMDD date."""
    response = requests.get(
        SCOREBOARD_URL, params={"dates": date_str}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    events: list[dict[str, Any]] = response.json().get("events", [])
    return [
        e["id"]
        for e in events
        if e.get("status", {}).get("type", {}).get("description") == "Final"
    ]


def fetch_game_summary(event_id: str) -> dict[str, Any]:
    """Fetch the full game summary JSON for a single event ID."""
    response = requests.get(
        SUMMARY_URL, params={"event": event_id}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def extract_game_meta(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract game_id, date, home_team, away_team, home_score, away_score."""
    competition: dict[str, Any] = summary["header"]["competitions"][0]
    game_id: str = competition["id"]
    raw_date: str = competition["date"]
    game_date = raw_date[:10]  # "2025-04-07T00:00Z" → "2025-04-07"

    home_team = ""
    away_team = ""
    home_score = 0
    away_score = 0

    for competitor in competition["competitors"]:
        name: str = competitor["team"]["displayName"]
        score = int(competitor.get("score", 0) or 0)
        if competitor["homeAway"] == "home":
            home_team = name
            home_score = score
        else:
            away_team = name
            away_score = score

    return {
        "game_id": game_id,
        "date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
    }


def extract_top_performers(
    summary: dict[str, Any], n: int = 3
) -> list[Performer]:
    """Return the top n scorers across both teams."""
    try:
        all_athletes: list[Performer] = []
        for team_obj in summary["boxscore"]["players"]:
            stats_block: dict[str, Any] = team_obj["statistics"][0]
            keys: list[str] = stats_block["keys"]
            pts_idx = keys.index("points")
            reb_idx = keys.index("rebounds")
            ast_idx = keys.index("assists")

            for entry in stats_block["athletes"]:
                if entry.get("didNotPlay", False):
                    continue
                stats: list[str] = entry.get("stats", [])
                if not stats:
                    continue
                try:
                    pts = int(stats[pts_idx] or 0)
                    reb = int(stats[reb_idx] or 0)
                    ast = int(stats[ast_idx] or 0)
                except (ValueError, IndexError):
                    continue
                all_athletes.append(
                    Performer(
                        player=entry["athlete"]["displayName"],
                        pts=pts,
                        reb=reb,
                        ast=ast,
                    )
                )

        all_athletes.sort(key=lambda x: x.pts, reverse=True)
        return all_athletes[:n]
    except (KeyError, IndexError, ValueError):
        return []


def extract_recap_text(summary: dict[str, Any]) -> str:
    """Extract and clean the game recap as plain text. Returns '' if unavailable."""
    try:
        raw_html: str = summary["article"]["story"]
        if not raw_html:
            return ""
        text = re.sub(r"<[^>]+>", "", raw_html)
        return " ".join(text.split())
    except KeyError:
        return ""


def build_stat_summary(performers: list[Performer]) -> str:
    """Build a templated stat string like 'Jayson Tatum: 31 pts, 11 reb, 5 ast'."""
    if not performers:
        return ""
    return "; ".join(
        f"{p.player}: {p.pts} pts, {p.reb} reb, {p.ast} ast" for p in performers
    )


def build_document(summary: dict[str, Any]) -> GameDocument:
    """Assemble and validate a GameDocument from a raw ESPN summary response."""
    meta = extract_game_meta(summary)
    performers = extract_top_performers(summary)
    recap = extract_recap_text(summary)
    stat_summary = build_stat_summary(performers)

    full_text_parts = [p for p in [recap, stat_summary] if p]
    full_text = " | ".join(full_text_parts)

    return GameDocument(
        **meta,
        top_performers=performers,
        recap_text=recap,
        stat_summary=stat_summary,
        full_text=full_text,
    )


def write_corpus(docs: list[GameDocument], output_path: Path) -> None:
    """Append documents to JSONL, one JSON object per line. Creates parent dirs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.model_dump_json() + "\n")
    print(f"Wrote {len(docs)} documents to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA game recaps from ESPN.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--days",
        type=int,
        default=LOOKBACK_DAYS,
        help="Number of past days to fetch games for (default: %(default)s)",
    )
    group.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format (requires --start-date)")
    args = parser.parse_args()

    dates: list[str] = []
    if args.start_date:
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date) if args.end_date else date.today()
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
    else:
        today = date.today()
        dates = [
            (today - timedelta(days=i)).strftime("%Y%m%d")
            for i in range(1, args.days + 1)
        ]

    all_game_ids: list[str] = []
    for date_str in dates:
        ids = fetch_game_ids_for_date(date_str)
        print(f"  {date_str}: {len(ids)} Final game(s)")
        all_game_ids.extend(ids)

    existing_game_ids = load_existing_ids(OUTPUT_PATH)
    game_ids_to_query = set(all_game_ids) - existing_game_ids
    print(f"\nGame summaries exist for {len(existing_game_ids)} games...\n")
    print(f"\nFetching summaries for {len(game_ids_to_query)} games...\n")
    docs: list[GameDocument] = []
    failed_ids: list[str] = []
    for game_id in game_ids_to_query:
        try:
            summary = fetch_game_summary(game_id)
            doc = build_document(summary)
            docs.append(doc)
            recap_status = "recap" if doc.recap_text else "no recap"
            print(
                f"  {doc.away_team} @ {doc.home_team} "
                f"— {doc.away_score}-{doc.home_score} "
                f"[{doc.date}, {recap_status}]"
            )
        except RequestException as e:
            logging.error(f"Network error fetching game {game_id}: {e}")
            failed_ids.append(game_id)
        except Exception as e:
            logging.error(f"Unexpected error processing game {game_id}: {e}")
            failed_ids.append(game_id)

    if failed_ids:
        print(f"\nFailed to fetch {len(failed_ids)} game(s): {failed_ids}")
    write_corpus(docs, OUTPUT_PATH)


if __name__ == "__main__":
    main()
