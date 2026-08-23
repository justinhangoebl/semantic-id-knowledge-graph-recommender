"""
SPRIGL post-processor and grammar logits processor.

LayeredSEMGrammarLogitsProcessor — enforces valid layered SEM block ordering
during beam search by masking illegal token choices at each generation step:

  Rule 1: after SEM_{k}_*  (k < N-1)  → only SEM_{k+1}_* tokens are valid
  Rule 2: after SEM_{N-1}_* (block done) → no SEM tokens are valid next
  Rule 3: after any non-SEM token       → SEM_1+…SEM_{N-1} are blocked
            (a new block can only start with SEM_0_*)

SPRIGLSequencePostProcessor — thin subclass of SPRIGSequencePostProcessor that
uses the same SEM-block scanning logic (startswith("SEM") still matches
SEM_0_42, SEM_1_17, …) and adds the grammar processor to the model.

CoveragePenaltyLogitsProcessor — soft, population-wide penalty used by SPRIGLC
to push beam search away from items that are already over-represented among
the beams/users being decoded together, directly targeting item coverage /
Gini index.
"""

import math
from collections import defaultdict

import torch
from transformers import LogitsProcessor

from hopwise.model.sprig_postprocessor import SPRIGSequencePostProcessor
from hopwise.utils import PathLanguageModelingTokenType


class LayeredSEMGrammarLogitsProcessor(LogitsProcessor):
    """Enforce valid SEM_{level}_{code} block grammar during generation."""

    def __init__(self, sem_token_ids_per_level: list, N: int, vocab_size: int):
        """
        Args:
            sem_token_ids_per_level: list of length N, each element a frozenset
                of token IDs belonging to that level (SEM_{l}_*).
            N: number of SEM tokens per item (= semantic_ids_per_item).
            vocab_size: total vocabulary size.
        """
        self.N = N
        self.vocab_size = vocab_size
        self.level_sets = [frozenset(s) for s in sem_token_ids_per_level]
        self.level_id_lists = [sorted(s) for s in sem_token_ids_per_level]

        all_sem: set = set()
        for ids in self.level_id_lists:
            all_sem.update(ids)
        self._all_sem_ids = sorted(all_sem)
        # IDs at levels 1..N-1 (can never start a new block)
        self._non_first_ids = sorted(all_sem - set(self.level_id_lists[0]))

        # Device-keyed cache for pre-computed boolean mask tensors.
        self._device_cache: dict = {}

    def _get_masks(self, device: torch.device) -> dict:
        key = str(device)
        if key in self._device_cache:
            return self._device_cache[key]

        V = self.vocab_size

        # block_if_at_level[l]: boolean tensor (V,), True = token must be blocked
        # when last token was SEM_{l}_* and l < N-1.
        # Everything except the next level (l+1) is blocked.
        block_if_at_level = []
        for l in range(self.N - 1):
            mask = torch.ones(V, dtype=torch.bool, device=device)
            next_ids = torch.tensor(self.level_id_lists[l + 1], dtype=torch.long, device=device)
            mask[next_ids] = False
            block_if_at_level.append(mask)

        # block_after_complete: block all SEM tokens when last was SEM_{N-1}_*
        block_after_complete = torch.zeros(V, dtype=torch.bool, device=device)
        if self._all_sem_ids:
            all_ids = torch.tensor(self._all_sem_ids, dtype=torch.long, device=device)
            block_after_complete[all_ids] = True

        # block_non_first: block SEM_1+ when last token was not a SEM token
        block_non_first = torch.zeros(V, dtype=torch.bool, device=device)
        if self._non_first_ids:
            nf_ids = torch.tensor(self._non_first_ids, dtype=torch.long, device=device)
            block_non_first[nf_ids] = True

        # Token-ID tensors for membership tests (which level is the last token)
        level_tensors = [
            torch.tensor(ids, dtype=torch.long, device=device)
            for ids in self.level_id_lists
        ]
        all_sem_tensor = torch.tensor(self._all_sem_ids, dtype=torch.long, device=device)

        masks = {
            "block_if_at_level": block_if_at_level,
            "block_after_complete": block_after_complete,
            "block_non_first": block_non_first,
            "level_tensors": level_tensors,
            "all_sem_tensor": all_sem_tensor,
        }
        self._device_cache[key] = masks
        return masks

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = scores.device
        masks = self._get_masks(device)
        last_tokens = input_ids[:, -1]  # (B,)

        is_any_sem = torch.isin(last_tokens, masks["all_sem_tensor"])

        # Rules 1 + 2: last token is a SEM token
        for l in range(self.N):
            is_l = torch.isin(last_tokens, masks["level_tensors"][l])
            if not is_l.any():
                continue

            if l < self.N - 1:
                # Rule 1: must continue with SEM_{l+1}_*; block everything else
                block = masks["block_if_at_level"][l]          # (V,)
                scores.masked_fill_(is_l.unsqueeze(1) & block.unsqueeze(0), float("-inf"))
            else:
                # Rule 2: block complete; no SEM token can follow
                scores.masked_fill_(
                    is_l.unsqueeze(1) & masks["block_after_complete"].unsqueeze(0),
                    float("-inf"),
                )

        # Rule 3: last token is NOT a SEM token; block mid-block SEM starters
        is_non_sem = ~is_any_sem
        if is_non_sem.any():
            scores.masked_fill_(
                is_non_sem.unsqueeze(1) & masks["block_non_first"].unsqueeze(0),
                float("-inf"),
            )

        return scores


class _SemBlockExclusionLogitsProcessor(LogitsProcessor):
    """Shared machinery for blocking SEM-block candidates whose full reachable
    item set is already covered by some "excluded" set of items.

    Builds ``children[depth][prefix] = {candidate_token: frozenset(item_ids
    reachable through prefix+candidate)}`` once from ``semantic_vocab``, then
    at each generation step blocks any candidate whose reachable set is a
    subset of ``self._excluded_items(seq, uid)`` — subclasses define what
    "excluded" means (in-path repetition vs. user history) by overriding that
    hook (and ``_uid_from_seq`` when the exclusion set is per-user).
    """

    def __init__(self, semantic_vocab, N: int, all_sem_ids):
        self.N = N
        self.semantic_vocab = semantic_vocab
        self._all_sem_ids = frozenset(int(t) for t in all_sem_ids)

        # children[depth][prefix] = {candidate_token: frozenset(item_ids reachable through prefix+candidate)}
        # prefix is a tuple of `depth` token IDs already chosen earlier in the current block.
        children: list[dict[tuple, dict[int, set]]] = [defaultdict(lambda: defaultdict(set)) for _ in range(N)]
        for item_id, tids in semantic_vocab.item_to_token_ids.items():
            tids = tuple(int(t) for t in tids[:N])
            for depth in range(len(tids)):
                children[depth][tids[:depth]][tids[depth]].add(int(item_id))
        self._children: list[dict[tuple, dict[int, frozenset]]] = [
            {prefix: {tok: frozenset(items) for tok, items in toks.items()} for prefix, toks in level.items()}
            for level in children
        ]

        self._all_sem_tensor_cache: dict = {}

    def _all_sem_tensor(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._all_sem_tensor_cache:
            self._all_sem_tensor_cache[key] = torch.tensor(
                sorted(self._all_sem_ids), dtype=torch.long, device=device
            )
        return self._all_sem_tensor_cache[key]

    def _scan_completed_items(self, token_ids: list[int]) -> set[int]:
        """Return the item IDs whose full N-token SEM block already appears in token_ids.

        A trailing, still-incomplete block (< N tokens) is naturally excluded since its
        run length never reaches N.
        """
        items: set[int] = set()
        i = 0
        n = len(token_ids)
        while i < n:
            if token_ids[i] in self._all_sem_ids:
                block = []
                j = i
                while j < n and token_ids[j] in self._all_sem_ids and len(block) < self.N:
                    block.append(token_ids[j])
                    j += 1
                if len(block) == self.N:
                    item_id = self.semantic_vocab.reverse_map(tuple(block))
                    if item_id is not None:
                        items.add(int(item_id))
                i = j
            else:
                i += 1
        return items

    def _block_prefix(self, token_ids: list[int]) -> tuple:
        """Trailing run of SEM tokens at the end of token_ids, capped at N (the
        in-progress block's prefix). Empty tuple if we're at a block boundary."""
        block = []
        for tok in reversed(token_ids):
            if tok in self._all_sem_ids and len(block) < self.N:
                block.append(tok)
            else:
                break
        block.reverse()
        return tuple(block)

    def _uid_from_seq(self, seq: list[int]) -> int | None:
        """Override when the exclusion set is per-user (default: not needed)."""
        return None

    def _excluded_items(self, seq: list[int], uid: int | None) -> frozenset[int]:
        raise NotImplementedError

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = scores.device

        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            prefix = self._block_prefix(seq)
            depth = len(prefix)
            if depth >= self.N:
                continue  # block just completed; nothing to modify at this position

            candidates = self._children[depth].get(prefix)
            if not candidates:
                continue

            uid = self._uid_from_seq(seq)
            excluded = self._excluded_items(seq, uid)
            if not excluded:
                continue

            to_block = [tok for tok, items in candidates.items() if items <= excluded]
            if to_block:
                scores[i, torch.tensor(to_block, dtype=torch.long, device=device)] = float("-inf")

        return scores


class PathSelfAvoidanceLogitsProcessor(_SemBlockExclusionLogitsProcessor):
    """Blocks the model from generating an item that already appears earlier in the
    same path being decoded.

    Checked at every position within a block, not just its start: given the SEM
    tokens already chosen earlier in the current block (the "prefix"), blocks any
    next-token candidate whose full set of items reachable through (prefix + candidate)
    has already been completely generated earlier in this specific sequence. At depth 0
    the reachable set behind a single level-0 token can be large (many items sharing a
    coarse code), so blocking rarely fires there; by the last position within a block,
    the prefix narrows the reachable set down to one (or a few, if collisions remain
    unresolved) items, so blocking becomes precise there.

    Unlike ``HistoryExclusionLogitsProcessor`` (which masks items from the user's past
    interactions), this depends only on what this beam has already produced during the
    current generation, so it directly targets in-path repetition rather than
    repetition of previously-seen items.
    """

    def _excluded_items(self, seq, uid):
        return self._scan_completed_items(seq)


class HistoryExclusionLogitsProcessor(_SemBlockExclusionLogitsProcessor):
    """Blocks a candidate SEM token when everything reachable through
    prefix+candidate is already in the user's training-history items.

    Checked at every block depth, not just depth 0 — as the prefix within a
    block fills in, the reachable item set shrinks (same mechanism
    ``PathSelfAvoidanceLogitsProcessor`` relies on for in-path repetition),
    so this catches cases the coarser depth-0-only history block in
    ``SPRIGSemanticPreferenceLogitsProcessor`` (used by the Boosted models)
    misses. Beams that would resolve to an already-seen item are discarded
    post-generation anyway (trainer-level history masking, plus
    ``SPRIGSequencePostProcessor.parse_sequences``'s own ``used_ids`` check) —
    redirecting them here instead gives the beam-search budget to novel items.
    """

    def __init__(self, semantic_vocab, used_ids, tokenizer, N: int, all_sem_ids):
        super().__init__(semantic_vocab, N, all_sem_ids)

        user_prefix = PathLanguageModelingTokenType.USER.token
        self._token_to_uid: dict[int, int] = {}
        for tok, tid in tokenizer.get_vocab().items():
            if tok.startswith(user_prefix):
                try:
                    self._token_to_uid[int(tid)] = int(tok[len(user_prefix):])
                except ValueError:
                    pass

        self._history: dict[int, frozenset[int]] = {
            uid: frozenset(int(x) for x in used_ids[uid])
            for uid in range(len(used_ids))
        }

    def _uid_from_seq(self, seq):
        # U_X token is always at position 1 in the sequence.
        return self._token_to_uid.get(seq[1] if len(seq) > 1 else -1)

    def _excluded_items(self, seq, uid):
        if uid is None:
            return frozenset()
        return self._history.get(uid, frozenset())


class CoveragePenaltyLogitsProcessor(_SemBlockExclusionLogitsProcessor):
    """Soft, population-wide penalty against SEM-block candidates whose reachable
    items are already over-represented among the beams being decoded together.

    ``PathSelfAvoidanceLogitsProcessor`` and ``HistoryExclusionLogitsProcessor``
    hard-block a candidate for one user/beam (in-path repetition, own history).
    This processor targets a different failure mode: SPRIGL's shared semantic
    codebook lets beam search across *many different users* converge on the
    same small set of high-probability codes, collapsing item coverage (the
    problem this class exists to fix — see the low ItemCoverage numbers that
    motivated it). Rather than blocking, it subtracts a soft penalty
    proportional to ``log1p(avg_count)``, where ``avg_count`` is how many rows
    in the *current decoding batch* (every user x beam being generated in this
    same ``generate()`` call, since HF beam search advances the whole batch in
    lockstep) have already completed each item reachable through a candidate.
    The tally is recomputed fresh from ``input_ids`` at every step — no
    persistent state — so it reflects only the batch actually being decoded
    together, a real (if batch-scoped) cross-user aggregate-diversity signal
    pushed into decoding instead of applied as post-hoc re-ranking.

    ``penalty_weight=0`` reproduces plain SPRIGL beam search exactly; larger
    values trade top-beam probability mass (and thus some ranking accuracy)
    for spreading recommendations across more of the catalog.
    """

    def __init__(self, semantic_vocab, N: int, all_sem_ids, penalty_weight: float = 1.0):
        super().__init__(semantic_vocab, N, all_sem_ids)
        self.penalty_weight = penalty_weight

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.penalty_weight <= 0:
            return scores

        rows = input_ids.tolist()

        # Population tally: how many rows in this decoding batch have already
        # completed each item, recomputed fresh every step (no persistent state).
        item_counts: dict[int, int] = defaultdict(int)
        for seq in rows:
            for item_id in self._scan_completed_items(seq):
                item_counts[item_id] += 1

        if not item_counts:
            return scores

        for i, seq in enumerate(rows):
            prefix = self._block_prefix(seq)
            depth = len(prefix)
            if depth >= self.N:
                continue  # block just completed; nothing to modify at this position

            candidates = self._children[depth].get(prefix)
            if not candidates:
                continue

            for tok, items in candidates.items():
                counts = [item_counts.get(it, 0) for it in items]
                if not any(counts):
                    continue
                avg_count = sum(counts) / len(counts)
                scores[i, tok] -= self.penalty_weight * math.log1p(avg_count)

        return scores

    def _excluded_items(self, seq, uid):
        # Unused: __call__ is fully overridden with a soft penalty instead of
        # the base class's hard subset-exclusion masking.
        raise NotImplementedError


class SPRIGLSequencePostProcessor(SPRIGSequencePostProcessor):
    """Post-processor for SPRIGL.

    Inherits all sequence scanning logic from SPRIGSequencePostProcessor
    unchanged — SEM_0_42, SEM_1_17, … all start with "SEM" so the existing
    startswith scan still detects them, and the reverse_map works via token IDs.

    The grammar logits processor is created here but attached to the model by
    SPRIGL.__init__ after the post-processor is constructed.
    """

    def __init__(
        self,
        tokenizer,
        used_ids,
        item_num,
        semantic_vocab,
        semantic_ids_per_item: int,
        topk=10,
        coverage_penalty_weight: float = 1.0,
    ):
        super().__init__(
            tokenizer=tokenizer,
            used_ids=used_ids,
            item_num=item_num,
            semantic_vocab=semantic_vocab,
            semantic_ids_per_item=semantic_ids_per_item,
            topk=topk,
        )
        N = int(semantic_ids_per_item)
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)

        # Build per-level token ID sets from the tokenizer vocabulary.
        # Tokens follow the pattern SEM_{level}_{code}.
        level_sets: list[set] = [set() for _ in range(N)]
        for tok, tid in vocab.items():
            if not tok.startswith("SEM_"):
                continue
            parts = tok.split("_")
            if len(parts) == 3:
                try:
                    l = int(parts[1])
                    if 0 <= l < N:
                        level_sets[l].add(tid)
                except ValueError:
                    pass

        self.grammar_processor = LayeredSEMGrammarLogitsProcessor(
            sem_token_ids_per_level=[frozenset(s) for s in level_sets],
            N=N,
            vocab_size=vocab_size,
        )

        self.path_self_avoidance_processor = PathSelfAvoidanceLogitsProcessor(
            semantic_vocab=semantic_vocab,
            N=N,
            all_sem_ids=self.sem_token_id_set,
        )

        self.history_exclusion_processor = HistoryExclusionLogitsProcessor(
            semantic_vocab=semantic_vocab,
            used_ids=used_ids,
            tokenizer=tokenizer,
            N=N,
            all_sem_ids=self.sem_token_id_set,
        )

        self.coverage_penalty_processor = CoveragePenaltyLogitsProcessor(
            semantic_vocab=semantic_vocab,
            N=N,
            all_sem_ids=self.sem_token_id_set,
            penalty_weight=coverage_penalty_weight,
        )
