"""
SPRIGTDataset — relation-free variant of SPRIGLDataset.

Items are still represented as N layered SEM tokens (SEM_0_<code>, SEM_1_<code>,
…), but no RELATION tokens are ever added to the vocabulary or emitted between
nodes when formatting a path. Sequences are shorter by exactly one token per
hop compared to SPRIGL.

The raw relation-interleaved path arrays produced by the inherited path
samplers (``generate_user_paths`` / ``_sample_pretrain_paths``) are left
untouched — they're still useful for graph-walk bookkeeping. Only
``_format_path`` stops reading the relation half of each array.
"""

import numpy as np

from hopwise.data.dataset.sprigl_dataset import SPRIGLDataset


class SPRIGTDataset(SPRIGLDataset):
    """SPRIGLDataset with relation tokens removed from vocab and formatted paths."""

    def _get_field_from_config(self):
        super()._get_field_from_config()

        # Parent formulas (SPRIGDataset._get_field_from_config) count exactly
        # one relation token per hop. Subtract it since SPRIGT emits none.
        if self.train_stage == "finetune":
            # parent: 2*H + 2*N + 1  ->  drop the H relation tokens
            self.token_sequence_length = self.path_hop_length + 2 * self.semantic_ids_per_item + 1
        else:
            # parent: (max_hop+1)*N + max_hop + 2  ->  drop the max_hop relation tokens
            max_hop = self.pretrain_hop_length[1]
            self.token_sequence_length = (max_hop + 1) * self.semantic_ids_per_item + 2

        self.context_length = self.token_sequence_length

    def _extra_vocab_segments(self) -> list[np.ndarray]:
        # No RELATION token block — SPRIGT never emits relation tokens.
        return []

    def _edge_relation_token(self, rel_id: int, vocab: dict) -> int:
        # No RELATION tokens in vocab to look up. get_tokenized_ckg's result is
        # only consumed by the generic KG-constrained-decoding logits
        # processor built (and immediately discarded) in
        # PathLanguageModelingRecommender.__init__ — SPRIG/SPRIGL/SPRIGT all
        # overwrite self.logits_processor_list right after with their own
        # SEM-token processors, so relation granularity here is never used.
        # Collapse every relation into a single placeholder bucket per head.
        return -1

    def _format_path(self, path: np.ndarray) -> str | None:
        """Convert a raw path array to a token string, dropping relations.

        Item nodes are expanded to N space-separated SEM tokens. Returns
        ``None`` if any item in the path lacks a semantic mapping.
        """
        path = path[path != self.PATH_PADDING]
        path_nodes = path[::2]  # relations at path[1::2] are intentionally ignored

        node_token_lists = self._path_node_token_lists(path_nodes)
        if node_token_lists is None:
            return None

        out: list[str] = []
        for node_toks in node_token_lists:
            out.extend(node_toks)

        return self.path_token_separator.join(out)
