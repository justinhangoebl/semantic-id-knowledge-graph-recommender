"""
SPRIGL — SPRIG with Layered Semantic IDs and grammar-constrained beam search.

Differences from SPRIG:
  1. Requires SPRIGLDataset (layered_semantic_ids=True), which builds a
     level-specific token namespace: SEM_0_<code>, SEM_1_<code>, …
  2. Adds LayeredSEMGrammarLogitsProcessor to the beam-search logits pipeline,
     enforcing that SEM blocks are always generated in the correct level order.
  3. Uses SPRIGLSequencePostProcessor for generation post-processing.
"""

from hopwise.model.path_language_modeling_recommender.sprig import SPRIG
from hopwise.model.sprigl_postprocessor import SPRIGLSequencePostProcessor


class SPRIGL(SPRIG):
    """SPRIG with layered SEM tokens and positional grammar masking.

    The grammar constraint enforces during beam search:
      - After SEM_{k}_* (k < N-1): next token must be SEM_{k+1}_*
      - After SEM_{N-1}_* (block complete): no SEM token may follow
      - After any non-SEM token: only SEM_0_* is a valid block start
    """

    def __init__(self, config, dataset):
        if not getattr(dataset, "layered_semantic_ids", False):
            raise ValueError(
                "SPRIGL requires a dataset with layered_semantic_ids=True. "
                "Use SPRIGLDataset (model: SPRIGL in config)."
            )

        super().__init__(config, dataset)

        # Replace the flat SPRIG post-processor with the layered-aware one.
        self.sequence_postprocessor = SPRIGLSequencePostProcessor(
            tokenizer=dataset.tokenizer,
            used_ids=dataset.get_user_used_ids(),
            item_num=dataset.item_num,
            semantic_vocab=self.semantic_vocab,
            semantic_ids_per_item=dataset.semantic_ids_per_item,
            topk=config["topk"],
        )

        # Attach the grammar and self-avoidance logits processors to the generation
        # pipeline. SPRIG.__init__ leaves logits_processor_list empty; we populate it here.
        self.logits_processor_list = [
            self.sequence_postprocessor.grammar_processor,
            self.sequence_postprocessor.path_self_avoidance_processor,
        ]
        if config["history_exclusion"]:
            self.logits_processor_list.append(self.sequence_postprocessor.history_exclusion_processor)
