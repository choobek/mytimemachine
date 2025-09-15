import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from training.utils.schedules import parse_step_schedule, value_for_step


def test_parse_empty_and_malformed():
    assert parse_step_schedule(None) == []
    assert parse_step_schedule("") == []
    # ignore stray commas and bad tokens
    s = parse_step_schedule(",,,0:0.1, 200:0.2, bad, 300:0.3 ")
    assert s == [(0, 0.1), (200, 0.2), (300, 0.3)]


def test_value_for_step_edges():
    sched = parse_step_schedule("100:0.5,200:0.7")
    # before first step -> None (no override yet)
    assert value_for_step(sched, 0) is None
    assert value_for_step(sched, 99) is None
    # at first and between steps
    assert value_for_step(sched, 100) == 0.5
    assert value_for_step(sched, 150) == 0.5
    assert value_for_step(sched, 200) == 0.7
    assert value_for_step(sched, 99999) == 0.7


