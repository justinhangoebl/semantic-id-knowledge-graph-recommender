"""
BoostedSPRIG / BoostedSPRIGL — SPRIG/SPRIGL with inference-time logits processors.

Adds SPRIGSemanticPreferenceLogitsProcessor on top of the base model:
  • History blocking  — at SEM block start, blocks first tokens that can only
                        reach items already in the user's history.
  • Semantic boost    — at each block depth k, adds a scaled prior toward the
                        SEM token IDs that appear at depth k across the user's
                        history items (cross-item generalisation signal).

No new parameters; weights are identical to SPRIG / SPRIGL.
Only the finetune stage needs to be run with model: BoostedSPRIG[L].

Config keys
-----------
sem_preference_alpha : float  weight of the preference prior (default 0.5).
"""

from hopwise.model.path_language_modeling_recommender.sprig import SPRIG
from hopwise.model.path_language_modeling_recommender.sprigl import SPRIGL
from hopwise.model.sprig_postprocessor import SPRIGSemanticPreferenceLogitsProcessor


def _build_preference_processor(model, config, dataset):
    return SPRIGSemanticPreferenceLogitsProcessor(
        semantic_vocab=model.semantic_vocab,
        used_ids=dataset.get_user_used_ids(),
        tokenizer=dataset.tokenizer,
        N=int(dataset.semantic_ids_per_item),
        vocab_size=len(dataset.tokenizer.get_vocab()),
        alpha=config["sem_preference_alpha"] or 0.5,
    )


class BoostedSPRIG(SPRIG):
    """SPRIG + semantic preference logits processor."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.logits_processor_list = [_build_preference_processor(self, config, dataset)]


class BoostedSPRIGL(SPRIGL):
    """SPRIGL + semantic preference logits processor.

    SPRIGL.__init__ already sets logits_processor_list = [grammar_processor].
    We append the preference processor so both run during beam search.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.logits_processor_list.append(_build_preference_processor(self, config, dataset))
