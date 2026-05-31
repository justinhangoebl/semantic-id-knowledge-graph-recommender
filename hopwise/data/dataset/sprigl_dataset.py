"""
SPRIGLDataset — layered semantic-ID variant of SPRIGDataset.

Items are represented as N consecutive SEM tokens with level-specific namespaces:
  SEM_0_<code>  SEM_1_<code>  ...  SEM_{N-1}_<code>

This makes hierarchy levels distinguishable to the language model and enables
the grammar-masking logits processor in SPRIGL that enforces valid block order
during beam search.

Config key ``layered_semantic_ids`` is always forced to True for this dataset;
all other config keys are inherited from SPRIGDataset.
"""

from hopwise.data.dataset.sprig_dataset import SPRIGDataset


class SPRIGLDataset(SPRIGDataset):
    """SPRIGDataset with layered semantic IDs (SPRIG-L).

    The only behavioural difference from SPRIGDataset is that
    ``layered_semantic_ids`` is always True regardless of the config value.
    """

    def _get_field_from_config(self):
        super()._get_field_from_config()
        self.layered_semantic_ids = True
