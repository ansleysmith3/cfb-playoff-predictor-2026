"""Build one row per (season, ranking week, team): that team's resume as of that moment.

Design rules
- NO LEAKAGE: a ranking labeled "week W" is released after week W-1's games, so only
  regular-season games from weeks <= W-1 are used. Postseason games are never used.
- RESUME ONLY: features come from game results and Elo ratings. Human polls (AP, Coaches)
  are kept as `ref_*` columns for comparison but must not be used as model inputs.
- NO TEAM NAMES as inputs: the model should learn how the committee turns a resume into
  a ranking, not "Alabama is usually good".

Writes data/features/features_<year>.csv

Usage:
    python -m src.features --years 2014-2026
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
FEATURE_DIR = ROOT / "data" / "features"

COMMITTEE = "Playoff Committee Rankings"

# FCS opponents have no Elo in the data; give them a fixed, clearly-below-average rating.
FCS_ELO = 1250.0
MARGIN_CAP = 28  # a 50-point blowout shouldn't count much more than a 28-point one
QUALITY_PCT = 0.80  # opponent in the top 20% of that week's Elo = a "quality" opponent
BAD_PCT = 0.50  # losing to an opponent below the median = a "bad" loss

# Games played after the committee's final list but still before the label's week.
# 2019: Army-Navy (Dec 14) shares CFBD week 15 with the conference championships.
DATE_CUTOFFS = {(2019, 16): "2019-12-08"}


def add_game_quality(games: pd.DataFrame) -> pd.DataFrame:
    """Add Elo-based opponent quality columns to the team-game table."""
    g = games.copy()
    g["opp_elo"] = g["opponent_pregame_elo"].where(g["opponent_is_fbs"], FCS_ELO)
    g["opp_elo"] = g["opp_elo"].fillna(FCS_ELO)

    # Percentile of this opponent's Elo among every FBS team playing that same week.
    def pct(week_df: pd.DataFrame) -> pd.Series:
        ref = np.sort(week_df["team_pregame_elo"].dropna().to_numpy())
        if len(ref) == 0:
            return pd.Series(np.nan, index=week_df.index)
        return pd.Series(np.searchsorted(ref, week_df["opp_elo"]) / len(ref), index=week_df.index)

    g["opp_pct"] = g.groupby("week", group_keys=False)[["team_pregame_elo", "opp_elo"]].apply(pct)
    g["margin_capped"] = g["margin"].clip(-MARGIN_CAP, MARGIN_CAP)
    g["quality_opp"] = g["opp_pct"] >= QUALITY_PCT
    g["bad_opp"] = g["opp_pct"] < BAD_PCT
    g["win"] = g["result"] == 1
    g["loss"] = ~g["win"]
    return g


def snapshot(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each team's games (already filtered to the cutoff) into resume features."""
    games = games.sort_values(["team", "week", "start_date"])
    by = games.groupby("team")

    out = pd.DataFrame(
        {
            "games": by.size(),
            "wins": by["win"].sum(),
            "losses": by["loss"].sum(),
            "fbs_wins": games[games["opponent_is_fbs"]].groupby("team")["win"].sum(),
            "fbs_losses": games[games["opponent_is_fbs"]].groupby("team")["loss"].sum(),
            "avg_margin": by["margin_capped"].mean(),
            "sos": by["opp_elo"].mean(),
            "quality_wins": games[games["win"] & games["quality_opp"]].groupby("team").size(),
            "quality_losses": games[games["loss"] & games["quality_opp"]].groupby("team").size(),
            "bad_losses": games[games["loss"] & games["bad_opp"]].groupby("team").size(),
            "best_win_elo": games[games["win"]].groupby("team")["opp_elo"].max(),
            "worst_loss_elo": games[games["loss"]].groupby("team")["opp_elo"].min(),
            "last3_wins": by["win"].apply(lambda s: s.tail(3).sum()),
            "elo_current": by["team_postgame_elo"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
            "conference": by["team_conference"].last(),
        }
    )
    count_cols = ["fbs_wins", "fbs_losses", "quality_wins", "quality_losses", "bad_losses"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["wins"] = out["wins"].astype(int)
    out["losses"] = out["losses"].astype(int)
    out["win_pct"] = out["wins"] / out["games"]
    out["undefeated"] = (out["losses"] == 0).astype(int)
    return out.reset_index()


def ranking_weeks(year: int, ranks: pd.DataFrame, games: pd.DataFrame) -> list[int]:
    """Which ranking weeks to build snapshots for."""
    committee = ranks[(ranks["poll"] == COMMITTEE) & (ranks["season_type"] == "regular")]
    if not committee.empty:
        return sorted(committee["week"].unique())
    # Season still in progress (no committee list yet): every AP week plus the next one.
    ap_weeks = set(ranks.loc[ranks["poll"] == "AP Top 25", "week"]) - {1}
    played = games[games["season_type"] == "regular"]["week"]
    return sorted(ap_weeks | {int(played.max()) + 1})


def build_year(year: int) -> pd.DataFrame:
    games = add_game_quality(pd.read_csv(CLEAN_DIR / f"team_games_{year}.csv"))
    ranks = pd.read_csv(CLEAN_DIR / f"rankings_{year}.csv").rename(columns={"seasonType": "season_type"})
    regular = games[games["season_type"] == "regular"]

    frames = []
    for week in ranking_weeks(year, ranks, games):
        cutoff = regular[regular["week"] <= week - 1]
        if (year, week) in DATE_CUTOFFS:
            cutoff = cutoff[cutoff["start_date"].str[:10] < DATE_CUTOFFS[(year, week)]]
        if cutoff.empty:
            continue

        snap = snapshot(cutoff)
        snap.insert(0, "ranking_week", week)
        snap.insert(0, "season", year)

        in_week = ranks[(ranks["week"] == week) & (ranks["season_type"] == "regular")]
        for poll, col in [(COMMITTEE, "committee_rank"), ("AP Top 25", "ref_ap_rank"), ("Coaches Poll", "ref_coaches_rank")]:
            lookup = in_week[in_week["poll"] == poll].set_index("school")["rank"]
            snap[col] = snap["team"].map(lookup)
        frames.append(snap)

    result = pd.concat(frames, ignore_index=True)
    result["in_committee_top12"] = (result["committee_rank"] <= 12).astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--years", help="Inclusive range such as 2014-2026")
    args = parser.parse_args()
    if args.years:
        first, last = (int(y) for y in args.years.split("-"))
        years = range(first, last + 1)
    else:
        years = [args.year]

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        df = build_year(year)
        df.to_csv(FEATURE_DIR / f"features_{year}.csv", index=False)
        labeled = df["committee_rank"].notna().sum()
        print(f"{year}: {len(df)} rows, {df['ranking_week'].nunique()} ranking weeks, {labeled} committee-labeled")


if __name__ == "__main__":
    main()
