import os
from typing import Optional, Union

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.trainer_utils import get_last_checkpoint

from hopwise.data import Interaction
from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM
from hopwise.model.sprig_postprocessor import SPRIGSequencePostProcessor
from hopwise.utils import PathLanguageModelingTokenType

class SPRIG(PEARLM):
    """SPRIG uses Semantic IDs: items are represented as N-token sequences (SEM{k} tokens)
    derived from hierarchical quantization, enabling cross-item generalization through
    shared code prefixes. This is the unconstrained ablation - no grammar masking.
    """

    TRAIN_STAGES = ["pretrain", "finetune"]

    def __init__(self, config, dataset):
        # Guard: SPRIG requires semantic_vocab from SPRIGDataset
        if not hasattr(dataset, "semantic_vocab") or dataset.semantic_vocab is None:
            raise ValueError(
                "SPRIG requires a dataset with semantic_vocab. "
                "Use SPRIGDataset with a valid .semanticids file."
            )

        config["context_length"] = dataset.token_sequence_length

        # SPRIG has a different token-type layout than PEARLM/KGGLM: it uses SEMANTIC tokens instead of ITEM tokens, requiring 4 type embeddings
        self.use_kg_token_types = config["use_kg_token_types"]
        if self.use_kg_token_types:
            raise ValueError("use_kg_token_types=True is not compatible with SPRIG's SEMANTIC token design. Set use_kg_token_types=False.")
        super().__init__(config, dataset)

        # Store semantic vocab for postprocessing
        self.semantic_vocab = dataset.semantic_vocab

        # Disable all logits processors
        self.logits_processor_list = []

        # Replace the default postprocessor with SPRIG's SEM-block scanner.
        self.sequence_postprocessor = SPRIGSequencePostProcessor(
            tokenizer=dataset.tokenizer,
            used_ids=dataset.get_user_used_ids(),
            item_num=dataset.item_num,
            semantic_vocab=self.semantic_vocab,
            semantic_ids_per_item=dataset.semantic_ids_per_item,
            topk=config["topk"],
        )

        self.train_stage = config["train_stage"]
        self.pre_model_path = config["pre_model_path"]
        self.semantic_ids_per_item = dataset.semantic_ids_per_item

        # Precompute SEM token ID tensor once for efficient terminal-mask construction.
        sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token
        sem_ids = sorted(
            tid for tok, tid in dataset.tokenizer.get_vocab().items()
            if tok.startswith(sem_prefix)
        )
        self.register_buffer(
            "_sem_token_ids",
            torch.tensor(sem_ids, dtype=torch.long),
            persistent=False,
        )

        # Label smoothing is required for both pretrain and finetune.
        # KGGLM does not need this because each item has a unique token.
        # SPRIG's SEM codes are shared across items (aliasing), so the model
        # can drive loss down by predicting popular codes rather than learning
        # item-specific representations. Smoothing prevents this collapse.
        self.loss = nn.CrossEntropyLoss()# no this is bad label_smoothing=0.1)
        self.post_init()


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

            # Adapt positional embeddings (wpe) when pretrain and finetune use
            # different sequence lengths (by design: each stage computes its own
            # token_sequence_length formula, so n_positions can legitimately differ).
            # We keep the first min(pretrain, finetune) position embeddings from the
            # pretrained checkpoint - exactly those that will be used during finetune -
            # and discard the rest.  This is intentional, not a workaround.
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

            # Adapt word token embeddings (wte) when pretrain and finetune have different vocab sizes. 
            wte_key = "transformer.wte.weight"
            if wte_key in weights:
                try:
                    current_wte = self.transformer.wte.weight
                except AttributeError:
                    current_wte = None

                pretrained_wte = weights[wte_key]
                if current_wte is not None and pretrained_wte.shape != current_wte.shape:
                    if pretrained_wte.shape[1] == current_wte.shape[1]:
                        # Same embedding dim, different vocab size.
                        min_vocab = min(pretrained_wte.shape[0], current_wte.shape[0])
                        new_wte = current_wte.clone()
                        new_wte[:min_vocab] = pretrained_wte[:min_vocab].to(current_wte.dtype)
                        weights[wte_key] = new_wte
                        self.logger.info(
                            f"WTE vocab mismatch: pretrain {pretrained_wte.shape[0]} vs "
                            f"finetune {current_wte.shape[0]} - copying {min_vocab} rows, "
                            f"reinitialising {current_wte.shape[0] - min_vocab} new token rows."
                        )
                    else:
                        # Embedding dimension changed; skip loading wte.
                        weights.pop(wte_key)

            self.load_state_dict(weights, strict=False)

    def _mask_labels_to_terminal(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Mask intermediate SEM blocks; keep terminal SEM block, relations, and entities.

        Only intermediate item SEM tokens are set to -100. Relation tokens and
        non-item entity tokens are kept — they carry unique KG structure signal
        with no aliasing risk, matching what KGGLM trains on.

        Gradient signal breakdown after masking (finetune, N=3, hop=3):
          - Terminal SEM block:       3 positions  (item target)
          - Relation tokens:          3 positions  (KG structure)
          - Non-item entity tokens:   0-3 positions (KG structure, path-dependent)
          - Intermediate SEM blocks:  masked        (aliasing risk suppressed)
          - BOS / EOS / user token:   masked        (specials, not informative)

        Uses a vectorised reverse-cumsum trick to identify the terminal N SEM tokens:
            rev_cumsum[b, t] = number of SEM tokens from position t to end of row.
        Positions where rev_cumsum <= N AND is_sem are the last N SEM positions.
        """
        is_sem = torch.isin(input_ids, self._sem_token_ids)          # (B, T)
        rev_cumsum = is_sem.float().flip(1).cumsum(1).flip(1)        # (B, T)
        terminal_sem = (rev_cumsum <= self.semantic_ids_per_item) & is_sem  # (B, T)

        # Intermediate SEM positions: SEM tokens that are NOT in the terminal block.
        intermediate_sem = is_sem & ~terminal_sem                    # (B, T)

        masked = labels.clone()
        masked[intermediate_sem] = -100
        return masked

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        if isinstance(input_ids, Interaction):
            token_type_ids = input_ids["token_type_ids"]
            attention_mask = input_ids["attention_mask"]
            input_ids = input_ids["input_ids"]

        if labels is not None and self.train_stage == "finetune":
            labels = self._mask_labels_to_terminal(input_ids, labels)

        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def explain(self, inputs, **kwargs):
        """Generate sequences and keep raw SEM tokens for semantic evaluation."""
        kwargs["max_length"] = self.token_sequence_length
        kwargs["min_length"] = self.token_sequence_length
        outputs = self.generate(inputs, **kwargs)

        max_new_tokens = self.token_sequence_length - inputs["input_ids"].size(1)
        scores, sequences = self.sequence_postprocessor.get_sequences(outputs, max_new_tokens=max_new_tokens)
        return scores, sequences

    def decode_path(self, path):
        """Standardize path format for SPRIG's variable-length SEM-token sequences.

        Unlike the base class which assumes fixed alternating node/relation
        positions and single-character prefixes, this handles multi-token SEM
        items and the ``SEM`` prefix (3 chars).
        """
        sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token
        user_prefix = PathLanguageModelingTokenType.USER.token
        relation_prefix = PathLanguageModelingTokenType.RELATION.token
        entity_prefix = PathLanguageModelingTokenType.ENTITY.token

        # Check BOS
        if path[0] != "[BOS]":
            raise ValueError("Path is not starting with a BOS Token")

        new_path = []
        i = 1  # skip [BOS] at index 0

        # In finetune, the first token after [BOS] is always the user token.
        if i < len(path) and path[i].startswith(user_prefix):
            user_id = int(path[i][len(user_prefix):])
            new_path.append(("self_loop", "user", user_id))
            i += 1
        elif self.train_stage == "finetune":
            raise ValueError("FINETUNE: paths must start with an user token")
        elif path[i].startswith(entity_prefix):
            entity_id = int(path[i][len(entity_prefix)])
            new_path.append(("self_loop", "user", entity_id))
            i += 1
        elif path[i].startswith(sem_prefix):
            sem_block = []
            while (
                i < len(path)
                and path[i].startswith(sem_prefix)
                and len(sem_block) < self.sequence_postprocessor.N
            ):
                sem_block.append(path[i])
                i += 1

            if len(sem_block) == self.sequence_postprocessor.N:
                # Reverse-map SEM token strings ->  token IDs -> item ID
                tok_ids = tuple(
                    self.sequence_postprocessor.tokenizer.convert_tokens_to_ids(t)
                    for t in sem_block
                )
                item_id = self.semantic_vocab.reverse_map(tok_ids)
                if item_id is not None:
                    new_path.append(("self_loop", "item", item_id))
                else:
                    new_path.append(("self_loop", "item", -1))
            else:
                # Incomplete SEM block
                new_path.append(("self_loop", "item", -1))

        # Remaining tokens: R <node_tokens> R <node_tokens> ...
        while i < len(path):
            tok = path[i]

            # Skip [EOS] / [PAD] / other specials
            if tok.startswith("["):
                break

            # Expect a relation token
            if not tok.startswith(relation_prefix):
                break
            relation = int(tok[len(relation_prefix):])
            i += 1

            if i >= len(path) or path[i].startswith("["):
                break

            # Collect the node: either a single U/E token or N consecutive SEM tokens.
            if path[i].startswith(user_prefix):
                uid = int(path[i][len(user_prefix):])
                new_path.append((relation, "user", uid))
                i += 1
            elif path[i].startswith(entity_prefix):
                eid = int(path[i][len(entity_prefix):])
                new_path.append((relation, "entity", eid))
                i += 1
            elif path[i].startswith(sem_prefix):
                # Collect up to N SEM tokens
                sem_block = []
                while (
                    i < len(path)
                    and path[i].startswith(sem_prefix)
                    and len(sem_block) < self.sequence_postprocessor.N
                ):
                    sem_block.append(path[i])
                    i += 1

                if len(sem_block) == self.sequence_postprocessor.N:
                    # Reverse-map SEM token strings -> token IDs -> item ID
                    tok_ids = tuple(
                        self.sequence_postprocessor.tokenizer.convert_tokens_to_ids(t)
                        for t in sem_block
                    )
                    item_id = self.semantic_vocab.reverse_map(tok_ids)
                    if item_id is not None:
                        new_path.append((relation, "item", item_id))
                    else:
                        new_path.append((relation, "item", -1))
                else:
                    new_path.append((relation, "item", -1))
            else:
                i += 1

        return new_path