

r"""SPRIG
##################################################

SPRIG is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
as paths extracted from a knowledge graph. It is trained to predict the next token in a sequence of tokens
representing a path. The model extends PEARLM by adding a self-supervised pre-training phase on a large corpus of paths extracted from the knowledge graph, which allows the model to learn better representations of entities and relations. The model can be used for explainable recommendation by generating paths that explain the recommendations made by the model.

"""

import os

from transformers.trainer_utils import get_last_checkpoint

from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM

class SPRIG(PEARLM):
    """SPRIG is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
    as paths extracted from a knowledge graph. It is trained to predict the next token in a sequence of tokens
    representing a path. The model extends PEARLM by adding a self-supervised pre-training phase on a large corpus of paths extracted from the knowledge graph, which allows the model to learn better representations of entities and relations. The model can be used for explainable recommendation by generating paths that explain the recommendations made by the model.

    SPRIG uses Semantic IDs: items are represented as N-token sequences (SEM{k} tokens)
    derived from hierarchical quantization, enabling cross-item generalization through
    shared code prefixes. This is the unconstrained ablation — no grammar masking.
    """

    TRAIN_STAGES = ["pretrain", "finetune"]

    def __init__(self, config, dataset):
        # Guard: SPRIG requires semantic_vocab from SPRIGDataset
        if not hasattr(dataset, "semantic_vocab") or dataset.semantic_vocab is None:
            raise ValueError(
                "SPRIG requires a dataset with semantic_vocab. "
                "Use SPRIGDataset with a valid .semanticids file."
            )

        # Override context_length to match SPRIG's variable-length sequences.
        # PEARLM reads config["context_length"] for n_positions and n_ctx,
        # but SPRIG sequences are longer due to multi-token items (N SEM tokens
        # per item vs 1 I token). The dataset computes the correct length.
        config["context_length"] = dataset.token_sequence_length

        super().__init__(config, dataset)

        # Store semantic vocab for postprocessing
        self.semantic_vocab = dataset.semantic_vocab

        # Disable all logits processors — SPRIG is the unconstrained ablation
        self.logits_processor_list = []

        self.train_stage = config["train_stage"]
        self.pre_model_path = config["pre_model_path"]

        assert self.train_stage in self.TRAIN_STAGES
        if self.train_stage == "finetune":
            # load pretrained model for finetune
            if not os.path.exists(os.path.join(self.pre_model_path, "config.json")):
                # if the path is not a checkpoint, we assume it contains the checkpoint
                self.pre_model_path = get_last_checkpoint(self.pre_model_path)

            if self.pre_model_path is None:
                raise ValueError(
                    "Could not find a valid checkpoint for finetuning. "
                    "Ensure pre_model_path points to a checkpoint or directory containing one."
                )

            from safetensors.torch import load_file

            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            weights = load_file(os.path.join(self.pre_model_path, "model.safetensors"))

            # Adapt positional embeddings (wpe) if context_length changed between
            # pretraining and finetuning. When the number of positions differs,
            # we either slice the pretrained matrix or fall back to the newly
            # initialized one so that load_state_dict does not raise a size error.
            wpe_key = "transformer.wpe.weight"
            if wpe_key in weights:
                try:
                    current_wpe = self.transformer.wpe.weight
                except AttributeError:
                    current_wpe = None

                pretrained_wpe = weights[wpe_key]
                if current_wpe is not None and pretrained_wpe.shape != current_wpe.shape:
                    if pretrained_wpe.shape[1] == current_wpe.shape[1]:
                        # Same embedding dim, different number of positions: copy
                        # as many positions as possible and keep the rest
                        # from the current (randomly initialised) matrix.
                        min_len = min(pretrained_wpe.shape[0], current_wpe.shape[0])
                        new_wpe = current_wpe.clone()
                        new_wpe[:min_len] = pretrained_wpe[:min_len]
                        weights[wpe_key] = new_wpe
                    else:
                        # Embedding dimension changed; skip loading wpe.
                        weights.pop(wpe_key)

            self.load_state_dict(weights, strict=False)