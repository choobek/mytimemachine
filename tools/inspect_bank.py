import argparse
import torch
from collections import OrderedDict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', type=str, required=True)
    p.add_argument('--actor-center', type=str, default=None)
    p.add_argument('--report-topk', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    bank = torch.load(a.in_path, map_location='cpu')
    bins = bank['bins']
    ages = bank['ages']
    total = sum(v.size(0) for v in bins.values())
    print(f'Total vectors: {total}')
    # per-bin counts
    ordered = OrderedDict(sorted(bins.items(), key=lambda kv: int(kv[0].split("-")[0])))
    for k, X in ordered.items():
        n = X.size(0)
        mean_age = float(ages[k].mean()) if n else float('nan')
        print(f'{k}: {n:6d} (mean age {mean_age:.1f})')

    # L2 norm check (sample)
    for k, X in ordered.items():
        if X.numel():
            norms = X[: min(1000, X.size(0))].norm(dim=1)
            print(f'{k}: L2 norms min/mean/max = {norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}')
            break

    if a.actor_center:
        mu = torch.load(a.actor_center, map_location='cpu').float()
        mu = torch.nn.functional.normalize(mu, dim=0)
        # sample global cosine stats
        sample_list = [X[: min(1000, X.size(0))] for X in ordered.values() if X.numel()]
        if len(sample_list) > 0:
            sample = torch.cat(sample_list, dim=0)
            cos = sample @ mu
            print(f'Cosine vs actor μ: mean {cos.mean():.3f}, std {cos.std():.3f}, max {cos.max():.3f}')
            if a.report_topk > 0:
                topv, _ = torch.topk(cos, k=min(a.report_topk, cos.numel()))
                print('Top cosines:', ' '.join(f'{v:.3f}' for v in topv.tolist()))


if __name__ == "__main__":
    main()


