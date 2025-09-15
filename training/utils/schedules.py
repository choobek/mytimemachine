def parse_step_schedule(spec: str):
    """
    Parse a simple step:value CSV schedule string into a sorted list of (step, value) pairs.
    Example: "0:0.05,20000:0.07,36000:0.05"
    Returns empty list if spec is falsy.
    """
    items = []
    if not spec:
        return items
    for tok in str(spec).split(","):
        tok = tok.strip()
        if len(tok) == 0:
            continue
        if ":" not in tok:
            continue
        step_s, val_s = tok.split(":", 1)
        try:
            items.append((int(step_s.strip()), float(val_s.strip())))
        except Exception:
            # ignore malformed entries
            continue
    items.sort(key=lambda x: x[0])
    return items


def value_for_step(schedule, step):
    """
    Given a sorted schedule list of (step, value) and a current step,
    return the last value whose step <= current step. If schedule is empty,
    return None. If the current step is before the first scheduled step,
    return None to indicate no override yet.
    """
    if not schedule:
        return None
    # Before first step — no value yet
    if int(step) < int(schedule[0][0]):
        return None
    cur = schedule[0][1]
    for s, v in schedule:
        if int(step) >= int(s):
            cur = v
        else:
            break
    return cur


