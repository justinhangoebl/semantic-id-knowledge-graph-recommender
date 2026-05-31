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
"""

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
