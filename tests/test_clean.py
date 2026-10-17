"""Tests for src/clean.py.

Unit tests use tiny hand-made frames. Data-quality tests check the real cleaned
output and are skipped if the pipeline hasn't been run yet.
"""
import pandas as pd
import pytest

from src.clean import CLEAN_DIR, POLLS_KEPT, flag_bad_scores, keep_fbs_games, to_team_games

YEAR = 2026


def make_games(**overrides) -> pd.DataFrame:
    row = {
        "id": 1, "season": 2026, "seasonType": "regular", "week": 1,
        "startDate": "2026-09-05T23:00:00.000Z", "completed": True,
        "neutralSite": False, "conferenceGame": False,
        "homeTeam": "Alpha", "homeClassification": "fbs", "homeConference": "X",
        "homePoints": 30.0, "homePregameElo": 1500.0,
        "awayTeam": "Beta", "awayClassification": "fbs", "awayConference": "Y",
        "awayPoints": 20.0, "awayPregameElo": 1400.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------- unit tests ----------

def test_missing_score_is_dropped_and_flagged():
    games = make_games(awayPoints=float("nan"))
    good, dropped = flag_bad_scores(games)
    assert good.empty
    assert dropped["drop_reason"].iloc[0] == "completed_but_missing_score"


def test_upcoming_game_without_score_is_not_flagged():
    games = make_games(completed=False, homePoints=float("nan"), awayPoints=float("nan"))
    good, dropped = flag_bad_scores(games)
    assert len(good) == 1 and dropped.empty


def test_fcs_only_game_is_removed_but_fbs_vs_fcs_is_kept():
    fcs_only = make_games(id=1, homeClassification="fcs", awayClassification="fcs")
    fbs_vs_fcs = make_games(id=2, awayClassification="fcs")
    kept = keep_fbs_games(pd.concat([fcs_only, fbs_vs_fcs]))
    assert kept["id"].tolist() == [2]


def test_team_games_has_two_rows_with_mirrored_results():
    rows = to_team_games(make_games())
    assert len(rows) == 2
    alpha = rows[rows["team"] == "Alpha"].iloc[0]
    beta = rows[rows["team"] == "Beta"].iloc[0]
    assert (alpha["result"], beta["result"]) == (1, 0)
    assert alpha["margin"] == 10 and beta["margin"] == -10
    assert alpha["is_home"] and not beta["is_home"]


def test_fcs_opponent_gets_one_row_and_is_flagged():
    rows = to_team_games(make_games(awayClassification="fcs"))
    assert rows["team"].tolist() == ["Alpha"]
    assert not rows["opponent_is_fbs"].iloc[0]


def test_neutral_site_is_not_home():
    rows = to_team_games(make_games(neutralSite=True))
    assert not rows["is_home"].any()


# ---------- data-quality checks on the real cleaned output ----------

@pytest.fixture(scope="module")
def team_games() -> pd.DataFrame:
    path = CLEAN_DIR / f"team_games_{YEAR}.csv"
    if not path.exists():
        pytest.skip("Run `python -m src.clean` first")
    return pd.read_csv(path)


def test_no_duplicate_team_game_rows(team_games):
    assert not team_games.duplicated(["game_id", "team"]).any()


def test_no_missing_scores(team_games):
    assert team_games[["points_for", "points_against"]].notna().all().all()


def test_scores_are_nonnegative(team_games):
    assert (team_games[["points_for", "points_against"]] >= 0).all().all()


def test_every_team_is_fbs(team_games):
    assert (team_games["team_classification"] == "fbs").all()


def test_no_fbs_ties(team_games):
    assert (team_games["margin"] != 0).all()


def test_team_plays_at_most_one_game_per_day(team_games):
    # Not "per week": CFBD labels Week 0 (late Aug) games as week 1, so a team
    # can legitimately have two week-1 games.
    day = team_games["start_date"].str[:10]
    assert not team_games.assign(day=day).duplicated(["team", "day"]).any()


def test_game_counts_are_plausible(team_games):
    assert team_games.groupby("team").size().max() <= 15


def test_win_loss_totals_balance_for_fbs_vs_fbs(team_games):
    both_fbs = team_games[team_games["opponent_is_fbs"]]
    assert both_fbs["result"].sum() == len(both_fbs) / 2


def test_ranked_schools_match_team_names(team_games):
    path = CLEAN_DIR / f"rankings_{YEAR}.csv"
    if not path.exists():
        pytest.skip("Rankings not cleaned yet")
    ranks = pd.read_csv(path)
    assert set(ranks["poll"]) <= set(POLLS_KEPT)
    unknown = set(ranks["school"]) - set(team_games["team"]) - set(team_games["opponent"])
    assert not unknown, f"Ranked schools missing from games data: {unknown}"
