import logging

import torch

from hopwise.model.sequence_postprocessor import BaseSequencePostProcessor
from hopwise.utils import PathLanguageModelingTokenType

logger = logging.getLogger(__name__)


class SPRIGSequencePostProcessor(BaseSequencePostProcessor):
    """Post-processor for SPRIG's semantic-ID sequences.
    """

    def __init__(self, tokenizer, used_ids, item_num, semantic_vocab, semantic_ids_per_item, topk=10):
        super().__init__(tokenizer, used_ids, item_num, topk=topk)
        self.semantic_vocab = semantic_vocab
        self.N = int(semantic_ids_per_item)

        # Pre-compute the set of all token IDs that are SEM tokens for O(1) membership tests.
        sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token
        vocab = tokenizer.get_vocab()
        self.sem_token_id_set = frozenset(
            tid for tok, tid in vocab.items() if tok.startswith(sem_prefix)
        )

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
        """
        user_num = user_index.unique().numel()
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = []

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

            # Extract all valid items from the sequence.
            items = self.get_recommended_items(token_ids)
            if not items:
                continue

            for item_id in items:
                item_id = int(item_id)
                if item_id < 0 or item_id >= self.item_num:
                    continue
                # First score wins (sequences are sorted by descending score).
                if torch.isfinite(scores[batch_uidx, item_id]):
                    continue
                if uid in self.used_ids and item_id in self.used_ids[uid]:
                    continue

                scores[batch_uidx, item_id] = sequence_score
                user_topk_sequences.append([uid, item_id, sequence_score.item(), seq_tokens])

        return scores, user_topk_sequences
    
    def get_recommended_items(self, token_ids, top_k=None):
        """Extract a ranked item list from a single generated token-ID sequence.

        Args:
            token_ids: Iterable of integer token IDs (one generated sequence).
            top_k: If given, return at most this many items.

        Returns:
            Deduplicated list of item IDs in order of first appearance.
        """
        token_ids = [int(t) for t in token_ids]
        candidates = []
        seen = set()

        i = 0
        while i < len(token_ids):
            if token_ids[i] in self.sem_token_id_set:
                # Try to collect exactly N consecutive SEM tokens.
                block = []
                j = i
                while j < len(token_ids) and token_ids[j] in self.sem_token_id_set and len(block) < self.N:
                    block.append(token_ids[j])
                    j += 1

                if len(block) == self.N:
                    item_id = self.semantic_vocab.reverse_map(tuple(block))
                    if (
                        item_id is not None
                        and 0 <= int(item_id) < self.item_num
                        and item_id not in seen
                    ):
                        candidates.append(int(item_id))
                        seen.add(int(item_id))
                    # Advance past the consumed block.
                    i = j
                else:
                    # Incomplete block — skip past whatever SEM tokens we found.
                    i = j
            else:
                i += 1

            if top_k is not None and len(candidates) >= top_k:
                break

        return candidates[:top_k] if top_k is not None else candidates

    def log_batch_diagnostics(self, batch_sequences):
        """Log diagnostic counters for a batch of generated sequences."""
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

        logger.info(
            "SPRIG batch diagnostics: "
            "valid_sem_block_rate=%.4f, known_item_rate=%.4f, "
            "known_item_seq_rate=%.4f, duplicate_item_rate=%.4f, "
            "mean_items_per_sequence=%.2f",
            valid_sem_block_rate,
            known_item_rate,
            known_item_seq_rate,
            duplicate_item_rate,
            mean_items_per_sequence,
        )

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