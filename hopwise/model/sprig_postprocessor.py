import logging
from collections import defaultdict

import torch
from transformers import LogitsProcessor

from hopwise.model.sequence_postprocessor import BaseSequencePostProcessor
from hopwise.utils import PathLanguageModelingTokenType

logger = logging.getLogger(__name__)


class SPRIGSemanticPreferenceLogitsProcessor(LogitsProcessor):
    """Two-in-one logits processor for SPRIG inference.

    History blocking (depth 0)
    --------------------------
    At the start of each SEM block, blocks first-SEM tokens whose entire
    reachable item set is already in the user's interaction history.  Those
    beams would be discarded post-generation anyway; redirecting them early
    gives the full beam budget to novel items.

    Semantic preference boosting (depth 0..N-2, NOT the final depth)
    ------------------------------------------------------------------
    At block position k < N-1, adds a scaled prior toward the SEM token IDs
    that appeared at depth k across the user's history items.  This injects a
    per-user "SEM code profile" — built solely from that user's own history,
    not from other users, so it is a personalization signal rather than a
    collaborative-filtering one.  It still generalises beyond the user's exact
    seen items, because the shared codebook means boosting a code also lifts
    other items that happen to share it, even ones this user never interacted
    with — that generalisation is across semantically similar *items* via
    code sharing, not across users.

    Deliberately excluded from the boost: depth N-1. By the final SEM
    position, the already-chosen prefix has narrowed the reachable-item set
    down to (near-)one specific item, so a preference prior there stops being
    "steer toward the user's taste neighborhood" and becomes "reward
    regenerating an item already in this user's history" — directly undoing
    the history-blocking half of this same processor, which exists precisely
    to keep beams away from fully-seen regions. Boosting only depths < N-1
    keeps the steering coarse (cluster/neighborhood-level) and leaves the
    fine-grained, item-identifying choice unbiased.

    Cross-user (collaborative) extension via code co-occurrence
    -------------------------------------------------------------
    The self-only profile above is a repetition signal, not a collaborative
    one — no other user's behavior enters it. ``beta`` blends in a genuine
    cross-user term: a population-level code-co-occurrence table (built once
    from every user's history, not just this one) is used to propagate this
    user's own code weights one hop through "codes that tend to co-occur with
    codes I already have, across everyone" (see ``_code_cooc`` / ``_cf_pref``).
    ``beta=0`` (default) recovers the exact single-user behavior above;
    higher ``beta`` weights the population signal more. Bounded by codebook
    size squared per depth, not catalog size, so it stays cheap regardless of
    how many items exist. Also depth-capped the same way as the self-only
    boost, for the same reason (never reach the item-identifying final SEM
    position), and filtered to drop any propagated code whose full reachable
    item set is already covered by this user's history (same criterion the
    depth-0 hard mask below already computes, generalised across depths via
    ``_tok_to_items_by_depth``) so it doesn't just re-derive "recommend what
    you've already seen" through a population detour.

    Args:
        semantic_vocab : SemanticVocab with item_to_token_ids populated.
        used_ids       : array-like, used_ids[uid] = set of interacted item IDs.
        tokenizer      : dataset tokenizer (to decode U_X token → uid integer).
        N              : semantic_ids_per_item.
        vocab_size     : total tokenizer vocabulary size.
        alpha          : weight of the self-only preference prior (default 0.5).
        beta           : weight of the cross-user co-occurrence term blended
                         into the self-only prior, 0..1 (default 0.0 — off,
                         exactly reproduces prior single-user-only behavior).
    """

    def __init__(
        self,
        semantic_vocab,
        used_ids,
        tokenizer,
        N: int,
        vocab_size: int,
        alpha: float = 0.5,
        beta: float = 0.0,
    ):
        self.N = N
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.beta = beta

        user_prefix = PathLanguageModelingTokenType.USER.token
        sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token

        # ── user token ID → uid integer ───────────────────────────────────────
        self._token_to_uid: dict[int, int] = {}
        for tok, tid in tokenizer.get_vocab().items():
            if tok.startswith(user_prefix):
                try:
                    self._token_to_uid[int(tid)] = int(tok[len(user_prefix):])
                except ValueError:
                    pass

        # ── all SEM token IDs ─────────────────────────────────────────────────
        self._all_sem: frozenset[int] = frozenset(
            int(tid) for tok, tid in tokenizer.get_vocab().items()
            if tok.startswith(sem_prefix)
        )

        # ── tok_to_items_by_depth[d][t] = frozenset of item IDs whose depth-d code is t
        tok_to_items_by_depth: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(N)]
        for item_id, tids in semantic_vocab.item_to_token_ids.items():
            for d, tid in enumerate(tids[:N]):
                tok_to_items_by_depth[d][int(tid)].add(int(item_id))
        self._tok_to_items_by_depth: list[dict[int, frozenset[int]]] = [
            {k: frozenset(v) for k, v in level.items()} for level in tok_to_items_by_depth
        ]
        # depth-0 view, kept under its old name for the hard-mask code below
        self._first_tok_to_items: dict[int, frozenset[int]] = self._tok_to_items_by_depth[0]

        # ── per-user item history (for history blocking) ──────────────────────
        self._history: dict[int, frozenset[int]] = {
            uid: frozenset(int(x) for x in used_ids[uid])
            for uid in range(len(used_ids))
        }

        # ── per-user SEM preference profile ──────────────────────────────────
        # _pref[uid][depth] = {sem_token_id: normalised_weight}
        self._pref: dict[int, list[dict[int, float]]] = {}
        for uid in range(len(used_ids)):
            depth_counts: list[defaultdict] = [defaultdict(float) for _ in range(N)]
            for item_id in used_ids[uid]:
                tids = semantic_vocab.item_to_token_ids.get(int(item_id))
                if tids is None:
                    continue
                for d, tid in enumerate(tids[:N]):
                    depth_counts[d][int(tid)] += 1.0
            depth_pref: list[dict[int, float]] = []
            for d in range(N):
                total = sum(depth_counts[d].values()) or 1.0
                depth_pref.append({k: v / total for k, v in depth_counts[d].items()})
            self._pref[uid] = depth_pref

        # ── population-level code co-occurrence (cross-user signal) ───────────
        # code_cooc[d][c1][c2] = count of times codes c1 and c2 (distinct) both
        # appeared at depth d across two different items in the SAME user's
        # history, summed over ALL users. Bounded by (codes-per-depth)^2, not
        # catalog size, since it's a code x code table, not an item x item one.
        code_cooc: list[defaultdict] = [defaultdict(lambda: defaultdict(float)) for _ in range(N)]
        if beta > 0.0:
            for uid in range(len(used_ids)):
                item_codes = [
                    semantic_vocab.item_to_token_ids[i][:N]
                    for i in used_ids[uid]
                    if i in semantic_vocab.item_to_token_ids
                ]
                for d in range(N):
                    codes_at_d = {int(tids[d]) for tids in item_codes}
                    for c1 in codes_at_d:
                        for c2 in codes_at_d:
                            if c1 != c2:
                                code_cooc[d][c1][c2] += 1.0
        self._code_cooc: list[dict[int, dict[int, float]]] = [
            {c1: dict(c2s) for c1, c2s in level.items()} for level in code_cooc
        ]

        # ── device cache: (uid, depth, device_str) → FloatTensor (V,) ────────
        self._pref_cache: dict = {}

    def _cf_pref(self, uid: int, depth: int) -> dict[int, float]:
        """One-step collaborative diffusion: propagate this user's own code
        weights through the population code-co-occurrence table, i.e. "codes
        that tend to co-occur (across everyone) with codes this user already
        has" — the cross-user counterpart to the self-only ``_pref`` above.
        """
        out: defaultdict = defaultdict(float)
        for c, w in self._pref[uid][depth].items():
            neighbors = self._code_cooc[depth].get(c)
            if not neighbors:
                continue
            total = sum(neighbors.values()) or 1.0
            for c2, count in neighbors.items():
                out[c2] += w * (count / total)
        return dict(out)

    def _pref_tensor(self, uid: int, depth: int, device: torch.device) -> torch.Tensor:
        key = (uid, depth, str(device))
        if key not in self._pref_cache:
            blended = self._pref[uid][depth]
            if self.beta > 0.0:
                history = self._history.get(uid, frozenset())
                cf_pref = {
                    c: w
                    for c, w in self._cf_pref(uid, depth).items()
                    # drop codes that would just re-derive "recommend an already-seen item"
                    if not (self._tok_to_items_by_depth[depth].get(c, frozenset()) <= history)
                }
                keys = set(blended) | set(cf_pref)
                blended = {
                    c: (1 - self.beta) * blended.get(c, 0.0) + self.beta * cf_pref.get(c, 0.0)
                    for c in keys
                }

            t = torch.zeros(self.vocab_size, dtype=torch.float32, device=device)
            for tok_id, weight in blended.items():
                if tok_id < self.vocab_size:
                    t[tok_id] = weight
            self._pref_cache[key] = t
        return self._pref_cache[key]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = scores.device

        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()

            # Count consecutive SEM tokens at end → depth within current block.
            block: list[int] = []
            for tok in reversed(seq):
                if tok in self._all_sem and len(block) < self.N:
                    block.append(tok)
                else:
                    break
            block.reverse()
            depth = len(block)

            if depth >= self.N:
                continue  # block just finished; nothing to modify

            # U_X token is always at position 1 in the sequence.
            uid = self._token_to_uid.get(seq[1] if len(seq) > 1 else -1)
            if uid is None:
                continue

            # ── semantic preference boost (coarse depths only, see class docstring) ──
            if depth < self.N - 1:
                scores[i] += self.alpha * self._pref_tensor(uid, depth, device)

            # ── history blocking (depth 0 only) ───────────────────────────────
            if depth == 0:
                history = self._history.get(uid, frozenset())
                to_block = [
                    t for t, reachable in self._first_tok_to_items.items()
                    if reachable <= history  # all reachable items already seen
                ]
                if to_block:
                    scores[i, torch.tensor(to_block, dtype=torch.long, device=device)] = float("-inf")

        return scores


class SPRIGSequencePostProcessor(BaseSequencePostProcessor):
    """Post-processor for SPRIG's semantic-ID sequences.
    """

    def __init__(self, tokenizer, used_ids, item_num, semantic_vocab, semantic_ids_per_item,
                 topk=10, held_out_items=None, cold_start_epsilon=1.0):
        super().__init__(tokenizer, used_ids, item_num, topk=topk)
        self.semantic_vocab = semantic_vocab
        self.N = int(semantic_ids_per_item)
        self.cold_start_epsilon = cold_start_epsilon

        # Pre-compute the set of all token IDs that are SEM tokens for O(1) membership tests.
        sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token
        vocab = tokenizer.get_vocab()
        self.sem_token_id_set = frozenset(
            tid for tok, tid in vocab.items() if tok.startswith(sem_prefix)
        )

        # TIGER-style prefix matching for cold-start.
        # Maps (t0, t1, …, t_{N-2}) → [cold_item_id, …] so that when the model
        # predicts a full SEM block for a seen item, held-out items sharing the
        # same first N-1 tokens are also surfaced as candidates.
        self._prefix_to_cold: dict[tuple, list[int]] = {}
        if held_out_items:
            for item_id in held_out_items:
                tids = semantic_vocab.item_to_token_ids.get(int(item_id))
                if tids is None or len(tids) < self.N:
                    continue
                prefix = tuple(tids[: self.N - 1])
                self._prefix_to_cold.setdefault(prefix, []).append(int(item_id))

    def get_sequences(self, generation_outputs, max_new_tokens=24):
        """Extract scores and structured paths from generation outputs.

        Uses beam-search sequence scores when available (PEARLM / SPRIG use
        HuggingFace beam search via ``GPT2LMHeadModel.generate``).
        """
        user_num = generation_outputs["sequences"][:, 1].unique().numel()

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(
            user_num, device=sequences.device
        ).repeat_interleave(num_return_sequences)

        sorted_indices = generation_outputs["sequences_scores"].argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]
        sorted_sequences_scores = generation_outputs["sequences_scores"][sorted_indices]

        self.log_batch_diagnostics(sorted_sequences)

        return self.parse_sequences(
            sorted_batch_user_index, sorted_sequences, sorted_sequences_scores
        )

    def parse_sequences(self, user_index, sequences, sequences_scores):
        """Parse generated sequences, extracting ALL valid items per sequence.

        Unlike the base class which extracts one item per sequence (the last
        token), SPRIG sequences contain multiple N-token SEM blocks along the
        path. Each valid block is scored with its parent sequence's beam score.
        Higher-scored beams are processed first (caller sorts descending), and
        the ``isfinite`` check ensures the first score seen for an item wins.

        When held_out_items were supplied at construction time, TIGER-style
        prefix matching is also applied: for each terminal SEM block that maps
        to a seen item, held-out items sharing the first N-1 tokens are added
        as additional candidates (with the same beam score).  cold_start_epsilon
        caps the fraction of cold candidates surfaced per user.
        """
        user_num = user_index.unique().numel()
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = []

        # Track cold-candidate count per user for ε enforcement.
        cold_counts: dict[int, int] = defaultdict(int)
        topk_max = max(self.topk) if isinstance(self.topk, (list, tuple)) else int(self.topk)
        cold_budget = int(topk_max * self.cold_start_epsilon)

        for batch_uidx, sequence, sequence_score in zip(user_index, sequences, sequences_scores):
            seq_tokens = self.tokenizer.decode(sequence).split(" ")
            token_ids = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)

            # Extract user ID from position after [BOS].
            if len(seq_tokens) < 2:
                continue
            uid_token = seq_tokens[1]
            if not uid_token.startswith(PathLanguageModelingTokenType.USER.token):
                continue
            uid = int(uid_token[len(PathLanguageModelingTokenType.USER.token):])

            # Extract terminal SEM block together with its raw token IDs.
            item_id, block_token_ids = self._get_terminal_item_and_block(token_ids)

            # ── exact match (seen item) ────────────────────────────────────────
            if item_id is not None:
                item_id = int(item_id)
                if 0 <= item_id < self.item_num:
                    if not torch.isfinite(scores[batch_uidx, item_id]):
                        if not (uid in self.used_ids and item_id in self.used_ids[uid]):
                            scores[batch_uidx, item_id] = sequence_score
                            user_topk_sequences.append(
                                [uid, item_id, sequence_score.item(), seq_tokens]
                            )

            # ── TIGER prefix matching (cold items) ────────────────────────────
            if block_token_ids is not None and self._prefix_to_cold:
                prefix = tuple(block_token_ids[: self.N - 1])
                cold_candidates = self._prefix_to_cold.get(prefix, [])
                u_idx = batch_uidx.item() if hasattr(batch_uidx, "item") else int(batch_uidx)
                for cold_id in cold_candidates:
                    if cold_counts[u_idx] >= cold_budget:
                        break
                    if cold_id < 0 or cold_id >= self.item_num:
                        continue
                    if torch.isfinite(scores[batch_uidx, cold_id]):
                        continue
                    if uid in self.used_ids and cold_id in self.used_ids[uid]:
                        continue
                    scores[batch_uidx, cold_id] = sequence_score
                    cold_counts[u_idx] += 1
                    user_topk_sequences.append(
                        [uid, cold_id, sequence_score.item(), seq_tokens]
                    )

        return scores, user_topk_sequences

    def _get_terminal_item_and_block(self, token_ids: list[int]):
        """Return (item_id, block_token_ids) for the last complete SEM block,
        or (None, None) if no complete block is found."""
        token_ids = [int(t) for t in token_ids]
        last_item = None
        last_block = None

        i = 0
        while i < len(token_ids):
            if token_ids[i] in self.sem_token_id_set:
                block = []
                j = i
                while j < len(token_ids) and token_ids[j] in self.sem_token_id_set and len(block) < self.N:
                    block.append(token_ids[j])
                    j += 1
                if len(block) == self.N:
                    item_id = self.semantic_vocab.reverse_map(tuple(block))
                    if item_id is not None and 0 <= int(item_id) < self.item_num:
                        last_item = int(item_id)
                        last_block = block
                    i = j
                else:
                    i = j
            else:
                i += 1
        return last_item, last_block
    
    def get_recommended_items(self, token_ids, top_k=None):
        """Extract the terminal item from a single generated token-ID sequence.

        Only the last complete N-token SEM block is returned, matching KGGLM's
        behaviour of scoring one item per beam (the terminal token). Intermediate
        SEM blocks are bridge-hops and receive no finetune gradient, so including
        them would dilute the user-conditioned signal with pretrain-quality noise.

        Args:
            token_ids: Iterable of integer token IDs (one generated sequence).
            top_k: Unused; kept for API compatibility.

        Returns:
            List with at most one item ID (the terminal SEM block), or [] if none.
        """
        token_ids = [int(t) for t in token_ids]
        last_item = None

        i = 0
        while i < len(token_ids):
            if token_ids[i] in self.sem_token_id_set:
                block = []
                j = i
                while j < len(token_ids) and token_ids[j] in self.sem_token_id_set and len(block) < self.N:
                    block.append(token_ids[j])
                    j += 1

                if len(block) == self.N:
                    item_id = self.semantic_vocab.reverse_map(tuple(block))
                    if item_id is not None and 0 <= int(item_id) < self.item_num:
                        last_item = int(item_id)
                    i = j
                else:
                    i = j
            else:
                i += 1

        return [last_item] if last_item is not None else []

    def log_batch_diagnostics(self, batch_sequences):
        """Log diagnostic counters for a batch of generated sequences.

        Scans ALL SEM blocks per sequence for diagnostics, but note that
        get_recommended_items() only returns the terminal (last) block.
        mean_items_per_sequence > 1 confirms bridge-hop items exist in paths
        but they are no longer used for recommendations.
        """
        if hasattr(batch_sequences, "tolist"):
            batch_sequences = batch_sequences.tolist()

        total_seqs = len(batch_sequences)
        if total_seqs == 0:
            return

        seqs_with_complete_block = 0
        seqs_with_item = 0
        total_complete_blocks = 0
        total_known_blocks = 0
        seqs_with_duplicates = 0
        total_distinct_items = 0

        for token_ids in batch_sequences:
            items, n_complete, n_known = self._scan_blocks_with_stats(token_ids)
            total_complete_blocks += n_complete
            total_known_blocks += n_known

            if n_complete > 0:
                seqs_with_complete_block += 1

            distinct = []
            seen = set()
            has_dup = False
            for item_id in items:
                if item_id in seen:
                    has_dup = True
                else:
                    seen.add(item_id)
                    distinct.append(item_id)

            if distinct:
                seqs_with_item += 1
            if has_dup:
                seqs_with_duplicates += 1
            total_distinct_items += len(distinct)

        valid_sem_block_rate = seqs_with_complete_block / total_seqs
        known_item_rate = total_known_blocks / total_complete_blocks if total_complete_blocks > 0 else 0.0
        known_item_seq_rate = seqs_with_item / total_seqs
        duplicate_item_rate = seqs_with_duplicates / total_seqs
        mean_items_per_sequence = total_distinct_items / total_seqs

        #logger.info(
        #    "SPRIG: "
        #    "valid_sem_block=%.4f, known_i_rate=%.4f, "
        #    "known_i_seq_rate=%.4f, dup_i_rate=%.4f, "
        #    "mean_i_per_seq=%.2f",
        #    valid_sem_block_rate,
        #    known_item_rate,
        #    known_item_seq_rate,
        #    duplicate_item_rate,
        #    mean_items_per_sequence,
        #)

        self._log_wandb_metrics(
            {
                "valid_sem_block_rate": valid_sem_block_rate,
                "known_item_rate": known_item_rate,
                "known_item_seq_rate": known_item_seq_rate,
                "duplicate_item_rate": duplicate_item_rate,
                "mean_items_per_sequence": mean_items_per_sequence,
            }
        )

    def _log_wandb_metrics(self, metrics):
        """Log diagnostics to W&B if a run is active."""
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({f"sprig/{k}": v for k, v in metrics.items()}, commit=False)
        except Exception:
            return

    def _scan_blocks_with_stats(self, token_ids):
        """Scan a token-ID sequence and return (items, n_complete_blocks, n_known_blocks)."""
        token_ids = [int(t) for t in token_ids]
        items = []
        n_complete = 0
        n_known = 0

        i = 0
        while i < len(token_ids):
            if token_ids[i] in self.sem_token_id_set:
                block = []
                j = i
                while j < len(token_ids) and token_ids[j] in self.sem_token_id_set and len(block) < self.N:
                    block.append(token_ids[j])
                    j += 1

                if len(block) == self.N:
                    n_complete += 1
                    item_id = self.semantic_vocab.reverse_map(tuple(block))
                    if item_id is not None:
                        n_known += 1
                        items.append(item_id)
                    i = j
                else:
                    i = j
            else:
                i += 1

        return items, n_complete, n_known