"""Turn a ranking into the actual 12-team playoff field, using the selection rules.

The committee ranks teams; the RULES decide who gets in. Rules differ by year:

  "2024"  5 highest-ranked conference champions get auto-bids. The 4 highest-ranked
          champions take seeds 1-4 (first-round byes); the other 8 are seeded by rank.
  "2025"  Same 5 auto-bids, but seeding is purely by rank (the top 4 ranked get byes).
  "2026"  Announced Aug 2026: auto-bids for the ACC, Big Ten, Big 12 and SEC champions,
          the highest-ranked champion from any other conference, and Notre Dame if it is
          ranked in the top 12. The rest of the 12 are at-large. The announcement did not
          describe seeding; we ASSUME seeding by rank as in 2025. Verify before relying on it.

Input is a table with columns: team, conference, conf_champion (0/1), rank (1 = best;
NaN = unranked). Output is the field, seed 1..12, with the reason each team got in.
"""
import pandas as pd

FIELD_SIZE = 12
P4_2026 = ["ACC", "Big Ten", "Big 12", "SEC"]
UNRANKED = 999  # champions outside the top 25 can still hold an auto-bid
RULES = ("2024", "2025", "2026")


def _auto_bids(teams: pd.DataFrame, rule: str) -> dict[str, str]:
    """Return {team: reason} for every automatic bid under the rule."""
    champs = teams[teams["conf_champion"] == 1]
    auto: dict[str, str] = {}

    if rule in ("2024", "2025"):
        for t in champs.sort_values("rank_sort").head(5)["team"]:
            auto[t] = "conference champion (top 5)"
        return auto

    for conf in P4_2026:
        c = champs[champs["conference"] == conf]
        if not c.empty:
            auto[c.iloc[0]["team"]] = f"{conf} champion"
    others = champs[~champs["conference"].isin(P4_2026 + ["FBS Independents"])]
    if not others.empty:
        auto[others.sort_values("rank_sort").iloc[0]["team"]] = "highest-ranked other champion"
    nd = teams[(teams["team"] == "Notre Dame") & (teams["rank_sort"] <= FIELD_SIZE)]
    if not nd.empty:
        auto.setdefault("Notre Dame", "Notre Dame (ranked top 12)")
    return auto


def select_field(teams: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    teams = teams.copy()
    teams["rank_sort"] = teams["rank"].fillna(UNRANKED)
    teams = teams.sort_values("rank_sort", kind="stable").reset_index(drop=True)

    auto = _auto_bids(teams, rule)
    at_large_slots = FIELD_SIZE - len(auto)
    at_large = [t for t in teams["team"] if t not in auto][:at_large_slots]

    field = teams[teams["team"].isin(list(auto) + at_large)].copy()
    field["reason"] = field["team"].map(auto).fillna("at-large")
    field["auto_bid"] = field["team"].isin(auto)

    if rule == "2024":
        # Byes go to the four highest-ranked CHAMPIONS, not the four highest-ranked teams.
        champs = field[field["conf_champion"] == 1].head(4)
        rest = field.drop(champs.index)
        field = pd.concat([champs, rest])
    field = field.reset_index(drop=True)
    field.insert(0, "seed", field.index + 1)
    return field[["seed", "team", "conference", "rank", "auto_bid", "reason"]]
