import re


MB_LINE_RE = re.compile(r"^mb:\s+prof=(?P<prof>\w+)\s+k=(?P<k>\d+)\s+band=\[(?P<band_min>[-+]?[0-9]*\.?[0-9]+),(?P<band_max>[-+]?[0-9]*\.?[0-9]+)\](?:\s+cand≈(?P<cand>[0-9]*\.?[0-9]+)\s+simμ≈(?P<sim_mean>[-+]?[0-9]*\.?[0-9]+)\s+p75≈(?P<p75>[-+]?[0-9]*\.?[0-9]+)\s+p90≈(?P<p90>[-+]?[0-9]*\.?[0-9]+))?")


def parse_mb_line(line: str):
    m = MB_LINE_RE.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    out = {
        'prof': d['prof'],
        'k': int(d['k']),
        'band_min': float(d['band_min']),
        'band_max': float(d['band_max']),
    }
    if d.get('cand') is not None:
        out.update({
            'candidate_count': float(d['cand']),
            'sim_mean': float(d['sim_mean']),
            'p75': float(d['p75']),
            'p90': float(d['p90']),
        })
    return out


def find_used_ema(line: str):
    # lines of the form 'used_ema=1 ...'
    if 'used_ema=' not in line:
        return None
    try:
        val = int(line.split('used_ema=')[1].split()[0].strip())
        return val
    except Exception:
        return None


def extract_k_effective(loss_dict_line: str):
    # loss dict repr may contain 'mb_k_effective': value
    try:
        if 'mb_k_effective' not in loss_dict_line:
            return None
        # crude parse for number after key
        idx = loss_dict_line.find('mb_k_effective')
        sub = loss_dict_line[idx:]
        # look for pattern like mb_k_effective': 32.0
        m = re.search(r"mb_k_effective['\"]?:\s*([0-9]*\.?[0-9]+)", sub)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None



