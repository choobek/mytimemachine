import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))
from training.utils.schedules import parse_step_schedule, value_for_step


def test_schedule_runtime_values():
	# Example schedule
	spec = "0:0.03,20000:0.05,36000:0.08"
	sched = parse_step_schedule(spec)
	# Exact boundary steps
	assert value_for_step(sched, 0) == 0.03
	assert value_for_step(sched, 20000) == 0.05
	assert value_for_step(sched, 36000) == 0.08
	# After last entry, value should hold
	assert value_for_step(sched, 50000) == 0.08

