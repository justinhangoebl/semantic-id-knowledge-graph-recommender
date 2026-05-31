"""Cold-start bucketed evaluation for SPRIG and KGGLM.

Loads a trained checkpoint and evaluates Recall@K / NDCG@K separately for
items grouped by how often they appeared in the training split.  This reveals
how performance degrades as item frequency decreases (cold-start behaviour).

Usage
-----
# ML-1M — existing 5-core models (buckets start at 5)
python cold_start_eval.py \
  --checkpoint ./saved/huggingface-distilgpt2-SPRIGL-May-30-2026_23-59-04.pth/checkpoint-17697 \
  --model SPRIGL \
  --dataset ml1m \
  --topk 10 20 \
  --gpu-id 4

# LFM — all-item models (buckets from 20)
python cold_start_eval.py \
  --checkpoint saved/SPRIG-lfm.pth \
  --model SPRIG \
  --dataset lfm \
  --config-files hopwise/properties/model/SPRIG.yaml \
  --topk 10 20 
  #

  
python cold_start_eval.py \
  --checkpoint ./saved/Pop-May-27-2026_19-04-05.pth \
  --model Pop \
  --dataset ml1m \
  --topk 10 20 \
  --buckets "0,4;5,9;10,19;20,49;50,9999"

python cold_start_eval.py \
  --checkpoint ./saved/SASRec-May-28-2026_00-18-13.pth \
  --model SASRec \
  --dataset ml1m \
  --topk 10 20 \
  --gpu-id 4 \
  --buckets "0,4;5,9;10,19;20,49;50,9999"
"""

import argparse
import math
import os
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


def hit_at_k(topk_hits: list[bool]) -> float:
    return 1.0 if any(topk_hits) else 0.0


def mrr_at_k(topk_hits: list[bool]) -> float:
    for r, h in enumerate(topk_hits):
        if h:
            return 1.0 / (r + 1)
    return 0.0


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
        bkt: (
            {f"recall@{k}": [] for k in topk_list}
            | {f"ndcg@{k}": [] for k in topk_list}
            | {f"hit@{k}": [] for k in topk_list}
            | {f"mrr@{k}": [] for k in topk_list}
        )
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
            if hasattr(model, "explain"):
                scores_raw, _ = model.explain(interaction, **path_gen_args)
            else:
                scores_raw = model.full_sort_predict(interaction)
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
                        bucket_metrics[bkt][f"hit@{k}"].append(hit_at_k(hits_k))
                        bucket_metrics[bkt][f"mrr@{k}"].append(mrr_at_k(hits_k))

    # Aggregate
    results = {}
    for bkt, metrics in bucket_metrics.items():
        agg: dict = {"users": len(metrics[f"recall@{topk_list[0]}"])}
        for metric, vals in metrics.items():
            agg[metric] = float(np.mean(vals)) if vals else 0.0
        results[bkt] = agg

    return results


# ---------------------------------------------------------------------------
# Percentile cold-start evaluation (held-out items)
# ---------------------------------------------------------------------------

def build_percentile_slices(
    held_out_freq: dict,
    n_items: int,
    pct_step: float = 1.0,
) -> list[tuple[float, float, set]]:
    """Divide held-out items into percentile slices of the full item set.

    Items are sorted by ascending pre-holdout frequency so the coldest items
    appear in the first slice.  Each slice spans `pct_step`% of all items.

    Returns list of (lo_pct, hi_pct, item_set) tuples.
    """
    sorted_items = sorted(held_out_freq.items(), key=lambda x: x[1])
    step = max(1, int(n_items * pct_step / 100))
    slices = []
    i = 0
    while i < len(sorted_items):
        chunk = sorted_items[i: i + step]
        lo_pct = i / n_items * 100
        hi_pct = (i + len(chunk)) / n_items * 100
        slices.append((lo_pct, hi_pct, {item for item, _ in chunk}))
        i += step
    return slices


def evaluate_cold_start(
    model,
    trainer,
    test_data,
    held_out_freq: dict,
    pct_step: float,
    topk_list: list[int],
    device: torch.device,
) -> dict:
    """Evaluate on held-out items grouped by frequency percentile slice.

    Returns
    -------
    results : dict  {(lo_pct, hi_pct): {"recall@K": ..., "hit@K": ...,
                                         "mrr@K": ..., "ndcg@K": ...,
                                         "coverage": ..., "users": int}}
              coverage = fraction of evaluated users where model generated ≥1
              held-out item (only meaningful for generative models).
    """
    path_gen_args = getattr(trainer, "path_generation_args", {})
    n_items = test_data._dataset.item_num
    max_k = max(topk_list)

    slices = build_percentile_slices(held_out_freq, n_items, pct_step)
    slice_masks = {
        (lo, hi): build_bucket_mask({i: 1 for i in items}, 1, 1, n_items)
        for lo, hi, items in slices
    }
    slice_keys = [(lo, hi) for lo, hi, _ in slices]

    metrics_init = lambda: (
        {f"recall@{k}": [] for k in topk_list}
        | {f"ndcg@{k}": [] for k in topk_list}
        | {f"hit@{k}": [] for k in topk_list}
        | {f"mrr@{k}": [] for k in topk_list}
        | {"covered": [], "users_total": []}
    )
    slice_metrics: dict = {key: metrics_init() for key in slice_keys}

    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(device)

            if hasattr(model, "explain"):
                scores_raw, _ = model.explain(interaction, **path_gen_args)
            else:
                scores_raw = model.full_sort_predict(interaction)
            scores = scores_raw.view(-1, n_items)

            scores[:, 0] = -torch.inf
            if history_index is not None:
                scores[history_index] = -torch.inf

            n_users_batch = scores.size(0)
            pos_matrix = torch.zeros(n_users_batch, n_items, dtype=torch.float32, device=device)
            pos_matrix[positive_u, positive_i] = 1.0

            for (lo, hi), mask in slice_masks.items():
                mask_d = mask.to(device)
                bkt_pos = pos_matrix * mask_d.unsqueeze(0)
                pos_len = bkt_pos.sum(dim=1)
                valid = pos_len > 0
                if not valid.any():
                    continue

                bkt_scores = scores.clone()
                bkt_scores[:, ~mask_d] = -torch.inf
                _, topk_idx = torch.topk(bkt_scores, min(max_k, n_items), dim=1)

                # coverage: did the model assign any finite score to a bucket item?
                has_finite = torch.isfinite(bkt_scores).any(dim=1)

                for u_batch in valid.nonzero(as_tuple=True)[0]:
                    u = u_batch.item()
                    n_pos = int(pos_len[u].item())
                    hits_all = bkt_pos[u][topk_idx[u]].bool().tolist()
                    covered = int(has_finite[u].item())

                    for k in topk_list:
                        hits_k = hits_all[:k]
                        slice_metrics[(lo, hi)][f"recall@{k}"].append(recall_at_k(hits_k, n_pos))
                        slice_metrics[(lo, hi)][f"ndcg@{k}"].append(ndcg_at_k(hits_k, n_pos))
                        slice_metrics[(lo, hi)][f"hit@{k}"].append(hit_at_k(hits_k))
                        slice_metrics[(lo, hi)][f"mrr@{k}"].append(mrr_at_k(hits_k))
                    slice_metrics[(lo, hi)]["covered"].append(covered)

    results = {}
    for key, m in slice_metrics.items():
        n = len(m["covered"])
        agg: dict = {"users": n}
        for metric, vals in m.items():
            if metric == "users_total":
                continue
            agg[metric] = float(np.mean(vals)) if vals else 0.0
        results[key] = agg
    return results


def print_cold_start_results(results: dict, topk_list: list[int], model_name: str, dataset: str):
    print(f"\nModel: {model_name}   Dataset: {dataset}   (cold-start held-out items by percentile)")
    cols = ["Slice".ljust(14), "Users".rjust(6), "Cover".rjust(7)]
    for k_ in topk_list:
        cols += [f"Hit@{k_}".rjust(8), f"Recall@{k_}".rjust(10),
                 f"NDCG@{k_}".rjust(9), f"MRR@{k_}".rjust(8)]
    print("  ".join(cols))
    print("─" * (14 + 6 + 7 + len(topk_list) * 37 + 4))
    for (lo, hi), agg in sorted(results.items()):
        label = f"[{lo:.1f}%,{hi:.1f}%]".ljust(14)
        row = [label, str(agg["users"]).rjust(6), f"{agg['covered']:.3f}".rjust(7)]
        for k_ in topk_list:
            row += [
                f"{agg[f'hit@{k_}']:.4f}".rjust(8),
                f"{agg[f'recall@{k_}']:.4f}".rjust(10),
                f"{agg[f'ndcg@{k_}']:.4f}".rjust(9),
                f"{agg[f'mrr@{k_}']:.4f}".rjust(8),
            ]
        print("  ".join(row))
    print()


# ---------------------------------------------------------------------------
# Pretty-print (frequency buckets)
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
# Checkpoint resolution (supports both .pth files and HuggingFace directories)
# ---------------------------------------------------------------------------

def resolve_checkpoint(checkpoint_path: str) -> tuple[str, "str | None"]:
    """Return (hopwise_pth, hf_checkpoint_dir_or_None).

    - Plain .pth file  → (checkpoint_path, None)
    - HF directory     → derives the companion hopwise-*.pth for config, returns
                         that as the first element and the HF dir as the second.
                         The directory can be the top-level huggingface-* dir or
                         a checkpoint-N subdirectory inside it.
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path, None

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine the huggingface-* top-level directory.
    # The user may pass either  huggingface-X/  or  huggingface-X/checkpoint-N/
    base = os.path.basename(checkpoint_path.rstrip("/"))
    if base.startswith("checkpoint-"):
        hf_dir = os.path.dirname(checkpoint_path.rstrip("/"))
        hf_checkpoint_dir = checkpoint_path
    else:
        hf_dir = checkpoint_path.rstrip("/")
        hf_checkpoint_dir = checkpoint_path

    hf_name = os.path.basename(hf_dir)
    if not hf_name.startswith("huggingface-"):
        raise ValueError(
            f"Cannot derive hopwise companion for '{checkpoint_path}'. "
            "Expected the directory to be named huggingface-* or contain a "
            "huggingface-* parent."
        )

    hopwise_name = "hopwise-" + hf_name[len("huggingface-"):]
    hopwise_path = os.path.join(os.path.dirname(hf_dir), hopwise_name)
    if not os.path.isfile(hopwise_path):
        raise FileNotFoundError(
            f"Expected companion hopwise config file not found: {hopwise_path}"
        )

    return hopwise_path, hf_checkpoint_dir


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
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID to use (overrides checkpoint config)")
    parser.add_argument(
        "--topk", nargs="+", type=int, default=[10, 20],
        help="K values for top-K metrics (default: 10 20)"
    )
    parser.add_argument(
        "--buckets", type=str,
        default="5,9;10,19;20,49;50,9999",
        help='Frequency buckets as "lo,hi;lo,hi;..." (default: ML-1M 5-core buckets). '
             'Ignored when the checkpoint was trained with cold_start_holdout_ratio.'
    )
    parser.add_argument(
        "--pct-step", type=float, default=1.0,
        help="Percentile slice width for cold-start held-out evaluation (default: 1.0%%)"
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")

    hopwise_pth, hf_checkpoint_dir = resolve_checkpoint(args.checkpoint)
    if hf_checkpoint_dir is not None:
        print(f"  HuggingFace checkpoint: {hf_checkpoint_dir}")
        print(f"  Config from:            {hopwise_pth}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=hopwise_pth,
    )

    gpu_id = args.gpu_id if args.gpu_id is not None else config["gpu_id"]
    use_gpu = (args.gpu_id is not None) or config["use_gpu"]
    device = torch.device(f"cuda:{gpu_id}" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if hf_checkpoint_dir is not None:
        from safetensors.torch import load_file
        weights = load_file(os.path.join(hf_checkpoint_dir, "model.safetensors"), device=str(device))
        model.load_state_dict(weights, strict=False)
        print(f"  Loaded weights from safetensors.")

    from hopwise.utils import get_trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Detect whether this is a cold-start holdout run or a frequency-bucket run.
    held_out_freq = getattr(train_data._dataset, "_held_out_item_freq", None)

    if held_out_freq:
        n_held_out = len(held_out_freq)
        n_items = train_data._dataset.item_num
        print(f"\nCold-start holdout detected: {n_held_out} items held out "
              f"({n_held_out / n_items:.1%} of {n_items} items).")
        print(f"  Pre-holdout freq range: "
              f"[{min(held_out_freq.values())}, {max(held_out_freq.values())}]")
        print(f"  Percentile step: {args.pct_step}%   TopK: {args.topk}")
        print(f"Running cold-start percentile evaluation ({len(test_data)} batches)...")
        results = evaluate_cold_start(
            model=model,
            trainer=trainer,
            test_data=test_data,
            held_out_freq=held_out_freq,
            pct_step=args.pct_step,
            topk_list=args.topk,
            device=device,
        )
        print_cold_start_results(results, args.topk, args.model, args.dataset)
    else:
        buckets = parse_buckets(args.buckets)
        print(f"\nFrequency-bucket mode.  Buckets: {buckets}   TopK: {args.topk}")
        item_train_freq = get_item_train_frequencies(train_data)
        freq_values = list(item_train_freq.values())
        print(f"  Items in training: {len(freq_values)}, "
              f"min={min(freq_values)}, max={max(freq_values)}, "
              f"median={sorted(freq_values)[len(freq_values)//2]}")
        print(f"Running bucketed evaluation ({len(test_data)} batches)...")
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
