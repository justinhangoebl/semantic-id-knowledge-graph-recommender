"""Cold-start bucketed evaluation for SPRIG and KGGLM.

Loads a trained checkpoint and evaluates Recall@K / NDCG@K separately for
items grouped by how often they appeared in the training split.  This reveals
how performance degrades as item frequency decreases (cold-start behaviour).

Usage
-----
python cold_start_eval.py \
  --checkpoint ./saved/huggingface-distilgpt2-BoostedSPRIGL-May-31-2026_18-22-27.pth/checkpoint-4719 \
  --model BoostedSPRIGL \
  --dataset ml1m \
  --gpu-id 1 \
  --topk 10 20 100 250 500 \
  --pct-step 5 \
  --epsilon 0.1 0.2 0.3 0.5 1.0


python cold_start_eval.py \
  --checkpoint ./saved/huggingface-distilgpt2-KGGLM-May-31-2026_14-48-53.pth/checkpoint-5899 \
  --model KGGLM \
  --dataset ml1m \
  --gpu-id 1 \
  --topk 10 20 100 250 500 \
  --pct-step 5 \
  --epsilon 1.0
  
  
python cold_start_eval.py \
  --checkpoint ./saved/huggingface-distilgpt2-BoostedSPRIGL-May-31-2026_18-44-50.pth/checkpoint-9988 \
  --model BoostedSPRIGL \
  --dataset lfm \
  --gpu-id 0 \
  --topk 10 20 100 250 500 \
  --pct-step 5 \
  --epsilon 0.1 0.2 0.3 0.5 1.0


python cold_start_eval.py \
  --checkpoint ./saved/huggingface-distilgpt2-KGGLM-May-31-2026_18-59-40.pth/checkpoint-12485 \
  --model KGGLM \
  --dataset lfm \
  --gpu-id 1 \
  --topk 10 20 100 250 500 \
  --pct-step 5 \
  --epsilon 1.0
  
python cold_start_eval.py \
  --checkpoint ./saved/SASRec-May-28-2026_01-59-26.pth \
  --model SASRec \
  --dataset lfm \
  --gpu-id 1 \
  --topk 10 20 100 250 500 \
  --pct-step 5 \
  --epsilon 1.0
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
    path_gen_args = dict(getattr(trainer, "path_generation_args", {}))
    # Diverse beam search (_group_beam_search) can trigger CUDA index errors in
    # the cold_start_eval batch loop; standard beam search is sufficient here.
    path_gen_args.pop("num_beam_groups", None)
    path_gen_args.pop("diversity_penalty", None)
    # Disable KV cache to avoid _reorder_cache CUDA index errors (HuggingFace
    # beam search).  BoostedSPRIGL's custom generate loop ignores this kwarg.
    path_gen_args["use_cache"] = False
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
    pct_step: float = 5.0,
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


def _apply_epsilon(scores: torch.Tensor, cold_mask: torch.Tensor,
                   epsilon: float, k: int) -> torch.Tensor:
    """Return a score tensor where cold items are capped at epsilon*k slots.

    For each user, if more than floor(epsilon*k) cold items appear in the
    natural top-k, the excess ones are set to -inf so they fall out of the
    ranking.  This matches TIGER's ε mechanism.
    """
    if epsilon >= 1.0:
        return scores
    cold_budget = max(0, int(k * epsilon))
    cold_indices = cold_mask.nonzero(as_tuple=False).squeeze(1).to(scores.device)
    if cold_indices.numel() == 0:
        return scores

    # Extract cold-item scores for all users at once: (n_users, n_cold)
    cold_scores = scores[:, cold_indices]
    out = scores.clone()

    for u in range(scores.size(0)):
        u_cold = cold_scores[u]                             # (n_cold,)
        finite = torch.isfinite(u_cold)
        if finite.sum() <= cold_budget:
            continue                                        # already within budget
        # Among finite cold scores, keep only the top cold_budget; suppress rest.
        finite_idx = finite.nonzero(as_tuple=False).squeeze(1)
        order = u_cold[finite_idx].argsort(descending=True)
        suppress_local = finite_idx[order[cold_budget:]]   # indices into cold_indices
        out[u, cold_indices[suppress_local]] = -torch.inf
    return out


def evaluate_cold_start(
    model,
    trainer,
    test_data,
    held_out_freq: dict,
    pct_step: float,
    topk_list: list[int],
    device: torch.device,
    epsilon_list=None,
) -> dict:
    """Evaluate held-out items for every epsilon in epsilon_list in one pass.

    Generation runs once (no ε cap).  Then for each ε the cold-item budget is
    applied at metric-computation time — no redundant GPU work.

    Returns
    -------
    results : dict  {(epsilon, lo_pct, hi_pct): {metric: value, "users": int}}
    """
    epsilon_list = epsilon_list or [1.0]
    path_gen_args = dict(getattr(trainer, "path_generation_args", {}))
    path_gen_args.pop("num_beam_groups", None)
    path_gen_args.pop("diversity_penalty", None)
    path_gen_args["use_cache"] = False
    n_items = test_data._dataset.item_num
    max_k = max(topk_list)

    # Temporarily disable ε cap so we collect all cold candidates.
    pp = getattr(model, "sequence_postprocessor", None)
    orig_eps = getattr(pp, "cold_start_epsilon", 1.0)
    if pp is not None:
        pp.cold_start_epsilon = 1.0

    slices = build_percentile_slices(held_out_freq, n_items, pct_step)
    slice_masks = {
        (lo, hi): build_bucket_mask({i: 1 for i in items}, 1, 1, n_items)
        for lo, hi, items in slices
    }

    # All held-out items as a single CPU mask for ε application.
    all_cold_mask = build_bucket_mask({i: 1 for i in held_out_freq}, 1, 1, n_items)

    def _metrics_init():
        return (
            {f"recall@{k}": [] for k in topk_list}
            | {f"ndcg@{k}": [] for k in topk_list}
            | {f"hit@{k}": [] for k in topk_list}
            | {f"mrr@{k}": [] for k in topk_list}
            | {"covered": []}
        )

    # Accumulate per (epsilon, slice_key).
    acc: dict = {
        (eps, lo, hi): _metrics_init()
        for eps in epsilon_list
        for lo, hi, _ in slices
    }

    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(device)

            if hasattr(model, "explain"):
                scores_raw, _ = model.explain(interaction, **path_gen_args)
            else:
                scores_raw = model.full_sort_predict(interaction)
            base_scores = scores_raw.view(-1, n_items)
            base_scores[:, 0] = -torch.inf
            if history_index is not None:
                base_scores[history_index] = -torch.inf

            n_users_batch = base_scores.size(0)
            pos_matrix = torch.zeros(n_users_batch, n_items, dtype=torch.float32, device=device)
            pos_matrix[positive_u, positive_i] = 1.0
            cold_mask_d = all_cold_mask.to(device)

            for eps in epsilon_list:
                # Apply ε budget to base scores.
                scores = _apply_epsilon(base_scores, cold_mask_d, eps, max_k)

                for (lo, hi), bkt_mask in slice_masks.items():
                    mask_d = bkt_mask.to(device)
                    bkt_pos = pos_matrix * mask_d.unsqueeze(0)
                    pos_len = bkt_pos.sum(dim=1)
                    valid = pos_len > 0
                    if not valid.any():
                        continue

                    bkt_scores = scores.clone()
                    bkt_scores[:, ~mask_d] = -torch.inf
                    _, topk_idx = torch.topk(bkt_scores, min(max_k, n_items), dim=1)
                    has_finite = torch.isfinite(bkt_scores).any(dim=1)

                    for u_batch in valid.nonzero(as_tuple=True)[0]:
                        u = u_batch.item()
                        n_pos = int(pos_len[u].item())
                        hits_all = bkt_pos[u][topk_idx[u]].bool().tolist()
                        covered = int(has_finite[u].item())
                        key = (eps, lo, hi)
                        for k in topk_list:
                            hits_k = hits_all[:k]
                            acc[key][f"recall@{k}"].append(recall_at_k(hits_k, n_pos))
                            acc[key][f"ndcg@{k}"].append(ndcg_at_k(hits_k, n_pos))
                            acc[key][f"hit@{k}"].append(hit_at_k(hits_k))
                            acc[key][f"mrr@{k}"].append(mrr_at_k(hits_k))
                        acc[key]["covered"].append(covered)

    # Restore original ε.
    if pp is not None:
        pp.cold_start_epsilon = orig_eps

    results = {}
    for key, m in acc.items():
        n = len(m["covered"])
        agg: dict = {"users": n}
        for metric, vals in m.items():
            agg[metric] = float(np.mean(vals)) if vals else 0.0
        results[key] = agg
    return results


def evaluate_random_cold_baseline(
    model,
    trainer,
    test_data,
    held_out_freq: dict,
    pct_step: float,
    topk_list: list[int],
    device: torch.device,
    epsilon_list=None,
    n_trials: int = 5,
) -> dict:
    """Random cold-start baseline: replace ε*K slots with random held-out items.

    The model's seen-item scores are kept; cold slots are filled by sampling
    uniformly from the slice's held-out items.  Averaged over n_trials for
    stability.  Same structure as evaluate_cold_start so results are directly
    comparable.
    """
    epsilon_list = epsilon_list or [1.0]
    path_gen_args = dict(getattr(trainer, "path_generation_args", {}))
    path_gen_args.pop("num_beam_groups", None)
    path_gen_args.pop("diversity_penalty", None)
    path_gen_args["use_cache"] = False
    n_items = test_data._dataset.item_num
    max_k = max(topk_list)

    # Run with ε=0 to get pure seen-item scores (no cold items in scores matrix).
    pp = getattr(model, "sequence_postprocessor", None)
    orig_eps = getattr(pp, "cold_start_epsilon", 1.0)
    if pp is not None:
        pp.cold_start_epsilon = 0.0

    slices = build_percentile_slices(held_out_freq, n_items, pct_step)
    slice_item_lists = {(lo, hi): sorted(items) for lo, hi, items in slices}
    slice_masks = {
        (lo, hi): build_bucket_mask({i: 1 for i in items}, 1, 1, n_items)
        for lo, hi, items in slices
    }

    def _metrics_init():
        return (
            {f"recall@{k}": [] for k in topk_list}
            | {f"ndcg@{k}": [] for k in topk_list}
            | {f"hit@{k}": [] for k in topk_list}
            | {f"mrr@{k}": [] for k in topk_list}
        )

    acc: dict = {
        (eps, lo, hi): _metrics_init()
        for eps in epsilon_list
        for lo, hi, _ in slices
    }

    rng = np.random.default_rng(42)

    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(device)

            if hasattr(model, "explain"):
                scores_raw, _ = model.explain(interaction, **path_gen_args)
            else:
                scores_raw = model.full_sort_predict(interaction)
            seen_scores = scores_raw.view(-1, n_items).clone()
            seen_scores[:, 0] = -torch.inf
            if history_index is not None:
                seen_scores[history_index] = -torch.inf

            n_users_batch = seen_scores.size(0)
            pos_matrix = torch.zeros(n_users_batch, n_items, dtype=torch.float32, device=device)
            pos_matrix[positive_u, positive_i] = 1.0

            for eps in epsilon_list:
                cold_budget = max(0, int(max_k * eps))

                for (lo, hi), bkt_mask in slice_masks.items():
                    mask_d = bkt_mask.to(device)
                    bkt_pos = pos_matrix * mask_d.unsqueeze(0)
                    pos_len = bkt_pos.sum(dim=1)
                    valid = pos_len > 0
                    if not valid.any():
                        continue

                    cold_pool = slice_item_lists[(lo, hi)]
                    if not cold_pool:
                        continue

                    for u_batch in valid.nonzero(as_tuple=True)[0]:
                        u = u_batch.item()
                        n_pos = int(pos_len[u].item())

                        trial_hits: dict = {f"{m}@{k}": [] for k in topk_list for m in ("recall", "ndcg", "hit", "mrr")}

                        for _ in range(n_trials):
                            # Build scores: seen items + random cold slots.
                            trial_scores = seen_scores[u].clone()
                            # Zero out all cold items first, then add random ones.
                            trial_scores[mask_d] = -torch.inf
                            n_rand = min(cold_budget, len(cold_pool))
                            rand_cold = rng.choice(cold_pool, size=n_rand, replace=False)
                            # Assign a score just below the seen-item minimum so
                            # random cold items fill the tail of the ranking.
                            finite_seen = trial_scores[torch.isfinite(trial_scores)]
                            cold_score = (finite_seen.min().item() - 1.0) if len(finite_seen) > 0 else -1.0
                            trial_scores[rand_cold] = cold_score

                            _, topk_idx = torch.topk(trial_scores, min(max_k, n_items))
                            hits_all = bkt_pos[u][topk_idx].bool().tolist()

                            for k in topk_list:
                                hits_k = hits_all[:k]
                                trial_hits[f"recall@{k}"].append(recall_at_k(hits_k, n_pos))
                                trial_hits[f"ndcg@{k}"].append(ndcg_at_k(hits_k, n_pos))
                                trial_hits[f"hit@{k}"].append(hit_at_k(hits_k))
                                trial_hits[f"mrr@{k}"].append(mrr_at_k(hits_k))

                        key = (eps, lo, hi)
                        for k in topk_list:
                            for m in ("recall", "ndcg", "hit", "mrr"):
                                acc[key][f"{m}@{k}"].append(float(np.mean(trial_hits[f"{m}@{k}"])))

    if pp is not None:
        pp.cold_start_epsilon = orig_eps

    results = {}
    for key, m in acc.items():
        n = len(m.get(f"recall@{topk_list[0]}", []))
        agg: dict = {"users": n}
        for metric, vals in m.items():
            agg[metric] = float(np.mean(vals)) if vals else 0.0
        results[key] = agg
    return results


def evaluate_pop_cold_baseline(
    test_data,
    held_out_freq: dict,
    pct_step: float,
    topk_list: list[int],
    device: torch.device,
) -> dict:
    """Popularity-based cold-start baseline (no model, no generation).

    Cold items are ranked by their pre-holdout training frequency — the most
    popular cold item ranks first for every user.  This is user-agnostic and
    runs in O(users × cold_items), no GPU needed.

    Beats this baseline ⟹ the model's cold-item signal (SEM prefix or entity
    token) carries information beyond raw popularity.

    Returns
    -------
    results : dict  {(1.0, lo_pct, hi_pct): {metric: value, "users": int}}
              ε is fixed at 1.0 (pure cold ranking, no seen-item mixing).
    """
    n_items = test_data._dataset.item_num
    max_k = max(topk_list)

    slices = build_percentile_slices(held_out_freq, n_items, pct_step)
    slice_masks = {
        (lo, hi): build_bucket_mask({i: 1 for i in items}, 1, 1, n_items)
        for lo, hi, items in slices
    }

    # Pre-build popularity score vector (same for every user).
    # Score = log(1 + pre_holdout_frequency) so the scale is comparable to
    # log-probability beam scores used by generative models.
    pop_scores_cpu = torch.full((n_items,), float("-inf"))
    for item_id, freq in held_out_freq.items():
        if item_id < n_items:
            pop_scores_cpu[item_id] = math.log1p(freq)

    def _metrics_init():
        return (
            {f"recall@{k}": [] for k in topk_list}
            | {f"ndcg@{k}": [] for k in topk_list}
            | {f"hit@{k}": [] for k in topk_list}
            | {f"mrr@{k}": [] for k in topk_list}
        )

    acc: dict = {(1.0, lo, hi): _metrics_init() for lo, hi, _ in slices}

    for batched_data in test_data:
        _, history_index, positive_u, positive_i = batched_data

        n_users_batch = int(positive_u.max().item()) + 1 if len(positive_u) > 0 else 0
        if n_users_batch == 0:
            continue

        pos_matrix = torch.zeros(n_users_batch, n_items, dtype=torch.float32)
        pos_matrix[positive_u, positive_i] = 1.0

        # Popularity scores are the same for every user in the batch.
        base = pop_scores_cpu.unsqueeze(0).expand(n_users_batch, -1).clone()

        # Mask items already seen by each user (history).
        if history_index is not None:
            base[history_index] = float("-inf")

        for (lo, hi), bkt_mask in slice_masks.items():
            bkt_pos = pos_matrix * bkt_mask.unsqueeze(0)
            pos_len = bkt_pos.sum(dim=1)
            valid = pos_len > 0
            if not valid.any():
                continue

            # Restrict scores to this slice's cold items only.
            bkt_scores = base.clone()
            bkt_scores[:, ~bkt_mask] = float("-inf")
            _, topk_idx = torch.topk(bkt_scores, min(max_k, n_items), dim=1)

            for u_batch in valid.nonzero(as_tuple=True)[0]:
                u = u_batch.item()
                n_pos = int(pos_len[u].item())
                hits_all = bkt_pos[u][topk_idx[u]].bool().tolist()
                key = (1.0, lo, hi)
                for k in topk_list:
                    hits_k = hits_all[:k]
                    acc[key][f"recall@{k}"].append(recall_at_k(hits_k, n_pos))
                    acc[key][f"ndcg@{k}"].append(ndcg_at_k(hits_k, n_pos))
                    acc[key][f"hit@{k}"].append(hit_at_k(hits_k))
                    acc[key][f"mrr@{k}"].append(mrr_at_k(hits_k))

    results = {}
    for key, m in acc.items():
        n = len(m.get(f"recall@{topk_list[0]}", []))
        agg: dict = {"users": n, "covered": 1.0}  # Pop-Cold always "covers" (has a score)
        for metric, vals in m.items():
            agg[metric] = float(np.mean(vals)) if vals else 0.0
        results[key] = agg
    return results


def print_cold_start_results(results: dict, topk_list: list[int], model_name: str,
                             dataset: str, label: str = ""):
    """Print multi-ε cold-start results.  results keys are (eps, lo_pct, hi_pct)."""
    all_keys = sorted(results.keys())
    epsilons = sorted({k[0] for k in all_keys})

    tag = f"  [{label}]" if label else ""
    for eps in epsilons:
        print(f"\nModel: {model_name}   Dataset: {dataset}{tag}   ε={eps:.2f}")
        cols = ["Slice".ljust(14), "Users".rjust(6), "Cover".rjust(7)]
        for k_ in topk_list:
            cols += [f"Hit@{k_}".rjust(8), f"Recall@{k_}".rjust(10),
                     f"NDCG@{k_}".rjust(9), f"MRR@{k_}".rjust(8)]
        print("  ".join(cols))
        print("─" * (14 + 6 + 7 + len(topk_list) * 37 + 4))
        for key in sorted(k for k in all_keys if k[0] == eps):
            _, lo, hi = key
            agg = results[key]
            slabel = f"[{lo:.1f}%,{hi:.1f}%]".ljust(14)
            cover = agg.get("covered", float("nan"))
            row = [slabel, str(agg["users"]).rjust(6), f"{cover:.3f}".rjust(7)]
            for k_ in topk_list:
                row += [
                    f"{agg.get(f'hit@{k_}', 0):.4f}".rjust(8),
                    f"{agg.get(f'recall@{k_}', 0):.4f}".rjust(10),
                    f"{agg.get(f'ndcg@{k_}', 0):.4f}".rjust(9),
                    f"{agg.get(f'mrr@{k_}', 0):.4f}".rjust(8),
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
        "--topk", nargs="+", type=int, default=[10, 20, 100, 200, 1000],
        help="K values for top-K metrics (default: 10 20 100)"
    )
    parser.add_argument(
        "--split", choices=["test", "valid", "both"], default="test",
        help="Which data split to evaluate: test (default), valid, or both (valid then test)"
    )
    parser.add_argument(
        "--buckets", type=str,
        default="5,9;10,19;20,49;50,9999",
        help='Frequency buckets as "lo,hi;lo,hi;..." (default: ML-1M 5-core buckets). '
             'Ignored when the checkpoint was trained with cold_start_holdout_ratio.'
    )
    parser.add_argument(
        "--pct-step", type=float, default=5.0,
        help="Percentile slice width for cold-start held-out evaluation (default: 5.0%%)"
    )
    parser.add_argument(
        "--epsilon", nargs="+", type=float, default=None,
        help="ε values to evaluate (default: 0.1 0.2 0.3 0.5 1.0). "
             "Each ε = max fraction of cold items in top-K."
    )
    parser.add_argument(
        "--random-baseline", action="store_true",
        help="Also run random cold-item baseline for comparison."
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

    epsilon_list = args.epsilon or [0.1, 0.2, 0.3, 0.5, 1.0]

    from hopwise.utils import get_trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    splits_to_run = (
        [("valid", valid_data), ("test", test_data)] if args.split == "both"
        else [("valid", valid_data)] if args.split == "valid"
        else [("test", test_data)]
    )

    # Detect whether this is a cold-start holdout run or a frequency-bucket run.
    held_out_freq = getattr(train_data._dataset, "_held_out_item_freq", None)

    if held_out_freq:
        n_held_out = len(held_out_freq)
        n_items = train_data._dataset.item_num
        print(f"\nCold-start holdout detected: {n_held_out} items held out "
              f"({n_held_out / n_items:.1%} of {n_items} items).")
        print(f"  Pre-holdout freq range: "
              f"[{min(held_out_freq.values())}, {max(held_out_freq.values())}]")

        # Ensure TIGER-style prefix lookup is built for SPRIG/SPRIGL models.
        # load_data_and_model passes the original unsplit dataset to the model
        # constructor, so _held_out_items may not have been set at init time.
        # Rebuild from train_data._dataset which always has the holdout metadata.
        pp = getattr(model, "sequence_postprocessor", None)
        held_out_items = getattr(train_data._dataset, "_held_out_items", None)
        if pp is not None and hasattr(pp, "_prefix_to_cold") and not pp._prefix_to_cold and held_out_items:
            for item_id in held_out_items:
                tids = pp.semantic_vocab.item_to_token_ids.get(int(item_id))
                if tids and len(tids) >= pp.N:
                    prefix = tuple(tids[: pp.N - 1])
                    pp._prefix_to_cold.setdefault(prefix, []).append(int(item_id))

        if pp is not None and getattr(pp, "_prefix_to_cold", None):
            n_prefixes = len(pp._prefix_to_cold)
            eps = pp.cold_start_epsilon
            print(f"  Prefix matching: ACTIVE — {n_prefixes} SEM prefixes → cold items  (ε={eps})")
        elif pp is not None:
            print(f"  Prefix matching: inactive (KGGLM/Pop/SASRec use direct token generation)")

        print(f"  Percentile step: {args.pct_step}%   TopK: {args.topk}   ε: {epsilon_list}")

        for split_name, split_data in splits_to_run:
            print(f"\n=== Split: {split_name.upper()} ({len(split_data)} batches) ===")
            results = evaluate_cold_start(
                model=model,
                trainer=trainer,
                test_data=split_data,
                held_out_freq=held_out_freq,
                pct_step=args.pct_step,
                topk_list=args.topk,
                device=device,
                epsilon_list=epsilon_list,
            )
            print_cold_start_results(results, args.topk, args.model, args.dataset,
                                     label=f"model/{split_name}")

            print(f"Running Pop-Cold baseline ({split_name})...")
            pop_results = evaluate_pop_cold_baseline(
                test_data=split_data,
                held_out_freq=held_out_freq,
                pct_step=args.pct_step,
                topk_list=args.topk,
                device=device,
            )
            print_cold_start_results(pop_results, args.topk, "Pop-Cold", args.dataset,
                                     label=f"popularity baseline/{split_name}")

            if args.random_baseline:
                print(f"Running random cold-start baseline ({split_name}, {len(split_data)} batches)...")
                rand_results = evaluate_random_cold_baseline(
                    model=model,
                    trainer=trainer,
                    test_data=split_data,
                    held_out_freq=held_out_freq,
                    pct_step=args.pct_step,
                    topk_list=args.topk,
                    device=device,
                    epsilon_list=epsilon_list,
                )
                print_cold_start_results(rand_results, args.topk, args.model, args.dataset,
                                         label=f"random baseline/{split_name}")
    else:
        buckets = parse_buckets(args.buckets)
        print(f"\nFrequency-bucket mode.  Buckets: {buckets}   TopK: {args.topk}")
        item_train_freq = get_item_train_frequencies(train_data)
        freq_values = list(item_train_freq.values())
        print(f"  Items in training: {len(freq_values)}, "
              f"min={min(freq_values)}, max={max(freq_values)}, "
              f"median={sorted(freq_values)[len(freq_values)//2]}")

        for split_name, split_data in splits_to_run:
            print(f"\n=== Split: {split_name.upper()} ({len(split_data)} batches) ===")
            results = evaluate_buckets(
                model=model,
                trainer=trainer,
                test_data=split_data,
                item_train_freq=item_train_freq,
                buckets=buckets,
                topk_list=args.topk,
                device=device,
            )
            print_results(results, args.topk, args.model, args.dataset)


if __name__ == "__main__":
    main()
