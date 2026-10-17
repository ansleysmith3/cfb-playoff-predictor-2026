"""Tests for src/report.py: the README must only change between the markers."""
import pandas as pd
import pytest

from src.report import END, START, render_block, replace_block


def make_field() -> pd.DataFrame:
    return pd.DataFrame(
        {"seed": [1, 2], "team": ["Indiana", "Georgia"], "conference": ["Big Ten", "SEC"],
         "record": ["6-0", "6-0"], "reason": ["Big Ten champion", "at-large"]}
    )


def test_block_contains_every_team_and_the_week():
    block = render_block(make_field(), week=8, updated="2026-10-20")
    assert block.startswith(START) and block.endswith(END)
    assert "| 1 | Indiana | Big Ten | 6-0 | Big Ten champion |" in block
    assert "ranking week 8" in block and "2026-10-20" in block


def test_only_the_marked_section_is_replaced():
    readme = f"before\n{START}\nold table\n{END}\nafter\n"
    new = replace_block(readme, render_block(make_field(), 9, "2026-10-27"))
    assert new.startswith("before\n") and new.endswith("\nafter\n")
    assert "old table" not in new and "Indiana" in new


def test_replacing_twice_does_not_duplicate_the_block():
    readme = f"{START}\nold\n{END}\n"
    once = replace_block(readme, render_block(make_field(), 8, "d"))
    twice = replace_block(once, render_block(make_field(), 9, "d"))
    assert twice.count(START) == 1 and twice.count(END) == 1


def test_missing_markers_raise_instead_of_silently_doing_nothing():
    with pytest.raises(ValueError):
        replace_block("no markers here", "block")
