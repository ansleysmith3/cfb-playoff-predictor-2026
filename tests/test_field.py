"""The rules engine must reproduce the REAL 2024 and 2025 playoff fields from the committee's
final rankings plus conference champions. If these pass, we can trust the same code for 2026."""
import pandas as pd
import pytest

from src.field import select_field
from src.model import FEATURE_DIR

# Sources: official CFP releases / news coverage of the Dec 8 2024 and Dec 7 2025 reveals.
REAL_2024 = ["Oregon", "Georgia", "Boise State", "Arizona State", "Texas", "Penn State",
             "Notre Dame", "Ohio State", "Tennessee", "Indiana", "SMU", "Clemson"]
REAL_2025_TOP4 = ["Indiana", "Ohio State", "Georgia", "Texas Tech"]
REAL_2025_FIELD = set(REAL_2025_TOP4) | {"Oregon", "Ole Miss", "Texas A&M", "Oklahoma",
                                         "Alabama", "Miami", "Tulane", "James Madison"}


def final_week(year: int) -> pd.DataFrame:
    f = pd.read_csv(FEATURE_DIR / f"features_{year}.csv")
    f = f[f["ranking_week"] == f["ranking_week"].max()]
    return f.rename(columns={"committee_rank": "rank"})[["team", "conference", "conf_champion", "rank"]]


def test_reproduces_real_2024_field_and_seeds():
    field = select_field(final_week(2024), "2024")
    assert field["team"].tolist() == REAL_2024


def test_reproduces_real_2025_field_and_byes():
    field = select_field(final_week(2025), "2025")
    assert set(field["team"]) == REAL_2025_FIELD
    assert field["team"].head(4).tolist() == REAL_2025_TOP4
    assert field.set_index("team").loc["Tulane", "seed"] == 11
    assert field.set_index("team").loc["James Madison", "seed"] == 12


def test_2025_notre_dame_was_left_out_under_2025_rules_but_gets_in_under_2026_rules():
    teams = final_week(2025)
    assert "Notre Dame" not in select_field(teams, "2025")["team"].tolist()
    # Under 2026 rules ND is ranked in the top 12 and gets an auto-bid (the FootballScoop example).
    field26 = select_field(teams, "2026")
    assert "Notre Dame" in field26["team"].tolist()
    assert "Duke" in field26["team"].tolist()  # ACC champion is automatic in 2026


def test_field_always_has_twelve_unique_teams():
    for year in (2017, 2019, 2022, 2023, 2024, 2025):
        for rule in ("2024", "2025", "2026"):
            field = select_field(final_week(year), rule)
            assert len(field) == 12 and field["team"].is_unique, (year, rule)
            assert field["seed"].tolist() == list(range(1, 13))


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        select_field(final_week(2025), "2030")
