import torch


def aggregate(name_to_tensor, expr: str, p_clean: torch.Tensor):
    import re
    def eval_func(token: str):
        token = token.strip()
        if token.startswith('mean(') and token.endswith(')'):
            inside = token[5:-1]
            names = [t.strip() for t in inside.split(',') if t.strip()]
            vals = [name_to_tensor[n] for n in names if n in name_to_tensor]
            return torch.stack(vals, dim=0).mean(dim=0) if len(vals)>0 else p_clean
        if token.startswith('min(') and token.endswith(')'):
            inside = token[4:-1]
            names = [t.strip() for t in inside.split(',') if t.strip()]
            vals = [name_to_tensor[n] for n in names if n in name_to_tensor]
            return (torch.stack(vals, dim=0).min(dim=0).values) if len(vals)>0 else p_clean
        return name_to_tensor.get(token, p_clean)
    sum_vec = torch.zeros_like(p_clean)
    for part in expr.split('+'):
        part = part.strip()
        m = re.match(r"^([0-9\.]+)\s*\*\s*(.+)$", part)
        if m:
            scale = float(m.group(1))
            func = m.group(2).strip()
            val = eval_func(func)
            sum_vec = sum_vec + scale * val
        else:
            val = eval_func(part)
            sum_vec = sum_vec + val
    return sum_vec


def test_tta_aggregator_math():
    B = 3
    clean = torch.tensor([0.8, 0.6, 0.4])
    flip = torch.tensor([0.7, 0.5, 0.3])
    jpeg = torch.tensor([0.6, 0.4, 0.2])
    blur = torch.tensor([0.55, 0.35, 0.1])
    name_to_tensor = {
        'clean': clean,
        'flip': flip,
        'jpeg75': jpeg,
        'blur0.6': blur,
    }
    expr = 'mean(clean,flip)+0.5*min(jpeg75,blur0.6)'
    out = aggregate(name_to_tensor, expr, p_clean=clean)
    expected = (clean + flip)/2.0 + 0.5*torch.min(jpeg, blur)
    assert torch.allclose(out, expected, atol=1e-8)

