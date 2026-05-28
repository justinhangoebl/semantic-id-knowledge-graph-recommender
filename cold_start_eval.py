"""Cold-start bucketed evaluation for SPRIG and KGGLM.

Loads a trained checkpoint and evaluates Recall@K / NDCG@K separately for
items grouped by how often they appeared in the training split.  This reveals
how performance degrades as item frequency decreases (cold-start behaviour).

Usage
-----
# ML-1M — existing 5-core models (buckets start at 5)
python cold_start_eval.py \\
  --checkpoint saved/SPRIG-ml1m.pth \\
  --model SPRIG \\
  --dataset ml1m \\
  --config-files hopwise/properties/model/SPRIG.yaml \\
  --topk 10 20 \\
  --buckets "5,9;10,19;20,49;50,9999"

# LFM — all-item models (buckets from 20)
python cold_start_eval.py \\
  --checkpoint saved/SPRIG-lfm.pth \\
  --model SPRIG \\
  --dataset lfm \\
  --config-files hopwise/properties/model/SPRIG.yaml \\
  --topk 10 20 \\
  --buckets "20,49;50,99;100,199;200,9999"
"""

import argparse
import math
from collections import Counter, defaultdict

import numpy as np
import torch

from hopwise.quick_start import load_data_and_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(topk_hits: list[bool], n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    return sum(topk_hits) / n_pos


def ndcg_at_k(topk_hits: list[bool], n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    dcg = sum(h / math.log2(r + 2) for r, h in enumerate(topk_hits))
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(n_pos, len(topk_hits))))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Training frequency computation
# ---------------------------------------------------------------------------

def get_item_train_frequencies(train_data) -> dict[int, int]:
    """Return {internal_item_id: count_in_training_split}."""
    dataset = train_data._dataset
    iid_field = dataset.iid_field
    inter = dataset.inter_feat
    return dict(Counter(inter[iid_field].numpy().tolist()))


# ---------------------------------------------------------------------------
# Bucket mask
# ---------------------------------------------------------------------------

def build_bucket_mask(item_train_freq: dict, lo: int, hi: int, n_items: int) -> torch.Tensor:
    """Boolean tensor of length n_items — True where training count in [lo, hi]."""
    mask = torch.zeros(n_items, dtype=torch.bool)
    for iid, cnt in item_train_freq.items():
        if iid < n_items and lo <= cnt <= hi:
            mask[iid] = True
    return mask


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_buckets(
    model,
    trainer,
    test_data,
    item_train_freq: dict,
    buckets: list[tuple[int, int]],
    topk_list: list[int],
    device: torch.device,
) -> dict:
    """
    Run beam-search on test_data and collect per-bucket Recall@K / NDCG@K.

    Returns
    -------
    results : dict  {(lo, hi): {"recall@K": float, "ndcg@K": float, "users": int}}
    """
    path_gen_args = getattr(trainer, "path_generation_args", {})
    n_items = test_data._dataset.item_num
    max_k = max(topk_list)

    # Per-bucket accumulators: bucket -> {metric@k: [per-user values]}
    bucket_metrics: dict = {
        bkt: {f"recall@{k}": [] for k in topk_list} | {f"ndcg@{k}": [] for k in topk_list}
        for bkt in buckets
    }

    # Pre-build bucket masks (on CPU; moved to device per batch)
    bucket_masks = {
        bkt: build_bucket_mask(item_train_freq, bkt[0], bkt[1], n_items)
        for bkt in buckets
    }

    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(device)

            # ---- get scores ------------------------------------------------
            # model.explain() returns (scores_tensor, paths)
            # scores shape: (n_users_batch, n_items)
            scores_raw, _paths = model.explain(interaction, **path_gen_args)
            scores = scores_raw.view(-1, n_items)

            # Mask PAD slot and seen history
            scores[:, 0] = -torch.inf
            if history_index is not None:
                scores[history_index] = -torch.inf

            n_users_batch = scores.size(0)

            # Build ground-truth matrix (batch_size × n_items)
            pos_matrix = torch.zeros(n_users_batch, n_items, dtype=torch.float32, device=device)
            pos_matrix[positive_u, positive_i] = 1.0

            # ---- per-bucket metrics ----------------------------------------
            for bkt, mask in bucket_masks.items():
                mask_d = mask.to(device)

                # Filter ground truth to bucket items only
                bkt_pos = pos_matrix * mask_d.unsqueeze(0)
                pos_len = bkt_pos.sum(dim=1)  # (n_users_batch,)

                # Only evaluate users who have ≥1 positive item in this bucket
                valid = pos_len > 0
                if not valid.any():
                    continue

                # Mask non-bucket items out of scores
                bkt_scores = scores.clone()
                bkt_scores[:, ~mask_d] = -torch.inf

                # Top-K per user
                _, topk_idx = torch.topk(bkt_scores, min(max_k, n_items), dim=1)

                for u_batch in valid.nonzero(as_tuple=True)[0]:
                    u = u_batch.item()
                    n_pos = int(pos_len[u].item())
                    hits_all = bkt_pos[u][topk_idx[u]].bool().tolist()

                    for k in topk_list:
                        hits_k = hits_all[:k]
                        bucket_metrics[bkt][f"recall@{k}"].append(recall_at_k(hits_k, n_pos))
                        bucket_metrics[bkt][f"ndcg@{k}"].append(ndcg_at_k(hits_k, n_pos))

    # Aggregate
    results = {}
    for bkt, metrics in bucket_metrics.items():
        agg: dict = {"users": len(metrics[f"recall@{topk_list[0]}"])}
        for metric, vals in metrics.items():
            agg[metric] = float(np.mean(vals)) if vals else 0.0
        results[bkt] = agg

    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_results(results: dict, topk_list: list[int], model_name: str, dataset: str):
    print(f"\nModel: {model_name}   Dataset: {dataset}")
    header_parts = ["Bucket".ljust(12), "Items (train freq)", "Users".rjust(7)]
    for k in topk_list:
        header_parts += [f"Recall@{k}".rjust(10), f"NDCG@{k}".rjust(10)]
    print("  ".join(header_parts))
    print("─" * (12 + 20 + 7 + len(topk_list) * 22 + 4))

    for (lo, hi), agg in sorted(results.items()):
        hi_str = str(hi) if hi < 9000 else "∞"
        bkt_label = f"[{lo},{hi_str}]".ljust(12)
        row = [bkt_label, f"{lo}–{hi_str} interactions".ljust(20), str(agg["users"]).rjust(7)]
        for k in topk_list:
            row += [
                f"{agg[f'recall@{k}']:.4f}".rjust(10),
                f"{agg[f'ndcg@{k}']:.4f}".rjust(10),
            ]
        print("  ".join(row))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_buckets(s: str) -> list[tuple[int, int]]:
    """Parse "5,9;10,19;20,49" → [(5,9),(10,19),(20,49)]."""
    buckets = []
    for part in s.split(";"):
        lo, hi = part.strip().split(",")
        buckets.append((int(lo), int(hi)))
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Cold-start bucketed evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pth checkpoint")
    parser.add_argument("--model", required=True, help="Model name (SPRIG or KGGLM)")
    parser.add_argument("--dataset", required=True, help="Dataset name (ml1m or lfm)")
    parser.add_argument(
        "--config-files", nargs="+", default=[],
        help="Additional config files (e.g. hopwise/properties/model/SPRIG.yaml)"
    )
    parser.add_argument(
        "--topk", nargs="+", type=int, default=[10, 20],
        help="K values for top-K metrics (default: 10 20)"
    )
    parser.add_argument(
        "--buckets", type=str,
        default="5,9;10,19;20,49;50,9999",
        help='Frequency buckets as "lo,hi;lo,hi;..." (default: ML-1M 5-core buckets)'
    )
    args = parser.parse_args()

    buckets = parse_buckets(args.buckets)
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Buckets: {buckets}   TopK: {args.topk}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.checkpoint,
        config_file_list=args.config_files if args.config_files else None,
    )

    device = torch.device(f"cuda:{config['gpu_id']}" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Build a minimal trainer just to reuse path_generation_args
    from hopwise.utils import get_trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    print(f"\nComputing item training frequencies from training split...")
    item_train_freq = get_item_train_frequencies(train_data)
    freq_values = list(item_train_freq.values())
    print(f"  Items in training: {len(freq_values)}, "
          f"min={min(freq_values)}, max={max(freq_values)}, "
          f"median={sorted(freq_values)[len(freq_values)//2]}")

    print(f"Running bucketed evaluation on test set ({len(test_data)} batches)...")
    results = evaluate_buckets(
        model=model,
        trainer=trainer,
        test_data=test_data,
        item_train_freq=item_train_freq,
        buckets=buckets,
        topk_list=args.topk,
        device=device,
    )

    print_results(results, args.topk, args.model, args.dataset)


if __name__ == "__main__":
    main()
