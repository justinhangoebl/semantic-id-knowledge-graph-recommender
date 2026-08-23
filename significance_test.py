"""
Wilcoxon signed-rank significance tests: SPRIGL (gen+spec) vs. baselines.
"""

import os
import numpy as np
from scipy.stats import wilcoxon

SCORES_DIR = "./sig_scores"
METRICS = ["ndcg@10", "mrr@10", "hit@10"]
DATASETS = ["ml1m", "onion"]  # must match config["dataset"] values

PROPOSED = "SPRIGL"
BASELINES = ["KGGLM"]


def load(metric, model, dataset):
    # metric name contains '@' which is valid in filenames
    path = os.path.join(SCORES_DIR, f"{metric}_{model}_{dataset}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def test(proposed, baseline, label):
    diff = proposed - baseline
    if np.all(diff == 0):
        print(f"    {label}: no difference")
        return
    stat, p = wilcoxon(proposed, baseline, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    direction = "better" if diff.mean() > 0 else "worse"
    print(f"    {label}: W={stat:.0f}, p={p:.4f} {sig}  ({direction}, Δmean={diff.mean():+.4f})")


print("Wilcoxon signed-rank tests (two-sided)")
print("* p<0.05  ** p<0.01  *** p<0.001  ns = not significant\n")

for dataset in DATASETS:
    print(f"=== {dataset} ===")
    for metric in METRICS:
        proposed_scores = load(metric, PROPOSED, dataset)
        if proposed_scores is None:
            print(f"  [{metric}] {PROPOSED} scores not found - run evaluate with HOPWISE_SAVE_USER_SCORES set")
            continue
        print(f"  [{metric}]  (n={len(proposed_scores)} users)")
        for baseline in BASELINES:
            baseline_scores = load(metric, baseline, dataset)
            if baseline_scores is None:
                print(f"    {PROPOSED} vs {baseline}: scores not found - skipped")
                continue
            if len(proposed_scores) != len(baseline_scores):
                print(f"    {PROPOSED} vs {baseline}: user count mismatch "
                      f"({len(proposed_scores)} vs {len(baseline_scores)}) - skipped")
                continue
            test(proposed_scores, baseline_scores, f"{PROPOSED} vs {baseline}")
    print()
