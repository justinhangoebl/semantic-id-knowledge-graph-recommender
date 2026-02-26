# @Time   : 2026/01
# @Author : Justin Hangoebl
# @Email  : hangoebl.j@gmail.com

r"""KnowledgePathSemanticIDDataset
##################################################
This dataset extends KnowledgePathDataset to support semantic IDs for items.
Instead of using simple item indices (ITEM0, ITEM1, ...), it uses semantic IDs
learned by an RQ-VAE model (SEM0, SEM1, ..., SEM255).
"""

import os

import numpy as np
import pandas as pd
from logging import getLogger


from hopwise.data.dataset import KnowledgePathDataset
from hopwise.utils import PathLanguageModelingTokenType, set_color


class KnowledgePathSemanticIDDataset(KnowledgePathDataset):
    """Dataset for path language modeling with semantic IDs.
    
    This class extends KnowledgePathDataset to use semantic IDs for item representation.
    It loads a .semanticids file that maps each item to a sequence of semantic token IDs.
    
    The semantic ID file should be a CSV with columns:
    - items: item IDs (0-indexed)
    - semantic_ids: list of semantic token IDs for each item
    
    Attributes:
        semantic_id_mapping (dict): Maps item IDs to their semantic ID sequences
        num_semantic_tokens (int): Total number of unique semantic tokens (e.g., 256)
        semantic_ids_per_item (int): Number of semantic tokens per item (e.g., 3)
    """
    
    def __init__(self, config):
        # Store semantic ID configuration before parent init
        self.num_semantic_tokens = config["num_semantic_tokens"]
        self.semantic_ids_per_item = config["semantic_ids_per_item"]
        self.semantic_id_mapping = None
        self.logger = getLogger()
        
        # Load semantic ID mapping before initializing tokenizer
        self._load_semantic_id_mapping(config)
        
        # Initialize parent class (this will call _get_field_from_config and _init_tokenizer)
        super().__init__(config)
        
        self.logger.info(
            set_color("Semantic ID Dataset", "blue") +
            f": {len(self.semantic_id_mapping) if self.semantic_id_mapping is not None else 0} items mapped to semantic IDs"
        )
    
    def _load_semantic_id_mapping(self, config):
        """Load the semantic ID mapping from the .semanticids file.
        
        The file should contain:
        - items: original item IDs
        - semantic_ids: list of semantic token IDs (as string representation of list)
        """
        dataset_name = config["dataset"]
        data_path = config["data_path"]
        semantic_id_file = os.path.join(data_path, f"{dataset_name}.semanticids")
        
        if not os.path.exists(semantic_id_file):
            raise FileNotFoundError(
                f"Semantic ID file not found: {semantic_id_file}\n"
                f"Please generate semantic IDs using RQ_Vae_Semantic_IDs.py first."
            )
        
        # Load the semantic ID mapping
        df = pd.read_csv(semantic_id_file)
        
        # Parse semantic_ids column (it's stored as string representation of list)
        # Example: "[1, 2, 69]" -> [1, 2, 69]
        import ast
        df['semantic_ids'] = df['semantic_ids'].apply(ast.literal_eval)
        
        # Create mapping: item_id -> semantic_id_list
        self.semantic_id_mapping = {}
        for _, row in df.iterrows():
            item_id = int(row['items'])
            semantic_ids = row['semantic_ids']
            self.semantic_id_mapping[item_id] = semantic_ids
        
        # Verify all items have the same number of semantic IDs
        semantic_id_lengths = [len(ids) for ids in self.semantic_id_mapping.values()]
        if len(set(semantic_id_lengths)) > 1:
            self.logger.warning(
                f"Items have different numbers of semantic IDs: {set(semantic_id_lengths)}"
            )
        else:
            self.semantic_ids_per_item = semantic_id_lengths[0]
        
        self.logger.debug(
            f"Loaded semantic IDs for {len(self.semantic_id_mapping)} items, "
            f"{self.semantic_ids_per_item} tokens per item, "
            f"{self.num_semantic_tokens} unique tokens"
        )
    
    def _init_tokenizer(self):
        """Initialize the tokenizer with semantic IDs for items.
        
        This overrides the parent method to use semantic tokens (SEM0, SEM1, ...)
        instead of item tokens (ITEM0, ITEM1, ...).
        """
        from tokenizers import Tokenizer, pre_tokenizers
        from tokenizers import models as token_models
        from tokenizers import processors as token_processors
        from tokenizers import trainers as token_trainers
        from transformers import PreTrainedTokenizerFast

        tokenizer_model_class = getattr(token_models, self.tokenizer_model)
        tokenizer_object = Tokenizer(tokenizer_model_class(unk_token=self.unk_token))
        
        # Pre-tokenizer definition
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split(self.path_token_separator, "removed")
        
        # Build vocabulary with semantic tokens instead of item tokens
        entity_range = np.arange(self.item_num, self.entity_num)
        
        # Create vocabulary:
        # - Users: U0, U1, ..., U{user_num-1}
        # - Semantic tokens: SEM0, SEM1, ..., SEM{num_semantic_tokens-1}
        # - Entities (non-items): E{item_num}, E{item_num+1}, ..., E{entity_num-1}
        # - Relations: R0, R1, ..., R{relation_num-1}
        token_vocab = np.concatenate([
            np.char.add(PathLanguageModelingTokenType.USER.token, np.arange(self.user_num).astype(str)),
            np.char.add(PathLanguageModelingTokenType.SEMANTIC.token, np.arange(self.num_semantic_tokens).astype(str)),
            np.char.add(PathLanguageModelingTokenType.ENTITY.token, entity_range.astype(str)),
            np.char.add(PathLanguageModelingTokenType.RELATION.token, np.arange(self.relation_num).astype(str)),
        ])
        
        # Train tokenizer
        tokenizer_trainer_class = getattr(token_trainers, self.tokenizer_model + "Trainer")
        tokenizer_trainer = tokenizer_trainer_class(
            vocab_size=len(token_vocab) + len(self.special_tokens),
            special_tokens=self.special_tokens
        )
        tokenizer_object.train_from_iterator(token_vocab, trainer=tokenizer_trainer)
        
        # Add post-processor for BOS and EOS
        tokenizer_object.post_processor = token_processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (spec_token, tokenizer_object.token_to_id(spec_token))
                for spec_token in [self.bos_token, self.eos_token]
            ],
        )
        
        # Create the PreTrainedTokenizerFast
        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            model_max_length=self.context_length,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token,
        )
    
    def _format_path(self, path):
        """Format a path as a string, converting item IDs to semantic ID sequences.
        
        This overrides the parent method to expand items into semantic token sequences.
        
        Args:
            path: Array of [user/entity/item IDs, relation IDs, ...]
        
        Returns:
            str: Formatted path with semantic tokens for items
        """
        # The path structure is: [entity, relation, entity, relation, ..., entity]
        # We need to identify which entities are items and expand them to semantic tokens
        
        formatted_tokens = []
        
        for i, id_value in enumerate(path):
            if id_value == self.PATH_PADDING:
                break
            
            # Determine if this is an entity or relation position
            is_entity = (i % 2 == 0)
            
            if is_entity:
                # Check if this is a user, item, or other entity
                if i == 0:
                    # First entity is always a user
                    formatted_tokens.append(f"{PathLanguageModelingTokenType.USER.token}{id_value}")
                elif id_value < self.user_num:
                    # This shouldn't happen in paths, but handle it
                    formatted_tokens.append(f"{PathLanguageModelingTokenType.USER.token}{id_value}")
                elif id_value < self.user_num + self.item_num:
                    # This is an item - convert to semantic tokens
                    item_id = id_value - self.user_num
                    if item_id in self.semantic_id_mapping:
                        semantic_ids = self.semantic_id_mapping[item_id]
                        # Add multiple semantic tokens for this item
                        for sem_id in semantic_ids:
                            formatted_tokens.append(f"{PathLanguageModelingTokenType.SEMANTIC.token}{sem_id}")
                    else:
                        # Fallback if item not in mapping
                        self.logger.warning(f"Item {item_id} not found in semantic ID mapping")
                        formatted_tokens.append(f"{PathLanguageModelingTokenType.ITEM.token}{item_id}")
                else:
                    # Other entity
                    formatted_tokens.append(f"{PathLanguageModelingTokenType.ENTITY.token}{id_value - self.user_num}")
            else:
                # Relation
                formatted_tokens.append(f"{PathLanguageModelingTokenType.RELATION.token}{id_value}")
        
        return self.path_token_separator.join(formatted_tokens)
    
    def get_tokenized_used_ids(self):
        """Convert the used ids to tokenized semantic ids.
        
        This overrides the parent method to use semantic tokens for items.
        
        Returns:
            dict: Maps tokenized user ids to sets of tokenized semantic item ids
        """
        user_token_type = PathLanguageModelingTokenType.USER.token
        
        used_ids = self.get_user_used_ids()
        tokenized_used_ids = {}
        
        for uid in range(used_ids.shape[0]):
            uid_token = self.tokenizer.convert_tokens_to_ids(user_token_type + str(uid))
            
            # For each item the user has interacted with, we need to add all possible
            # semantic token sequences that could represent that item
            semantic_item_sequences = set()
            for item_id in used_ids[uid]:
                if item_id in self.semantic_id_mapping:
                    semantic_ids = self.semantic_id_mapping[item_id]
                    # Add the first semantic token (for filtering during generation)
                    first_sem_token = f"{PathLanguageModelingTokenType.SEMANTIC.token}{semantic_ids[0]}"
                    semantic_item_sequences.add(self.tokenizer.convert_tokens_to_ids(first_sem_token))
            
            tokenized_used_ids[uid_token] = semantic_item_sequences
        
        return tokenized_used_ids
    
    def get_tokenized_ckg(self):
        """Return the tokenized collaborative knowledge graph with semantic IDs.
        
        This overrides the parent method to use semantic tokens for items in the graph.
        
        Returns:
            dict[dict[set]]: The tokenized collaborative knowledge graph
        """
        token_vocab = self.tokenizer.get_vocab()
        graph = self._create_ckg_igraph(show_relation=True, directed=False)
        vertex_metadata, edge_metadata = graph.to_dict_list()

        def igraph_id_to_tokenizer_id(igraph_head, igraph_relation, igraph_tail):
            """Convert igraph IDs to tokenizer IDs, handling semantic IDs for items."""
            ret = []
            triple = [igraph_head, igraph_relation, igraph_tail]
            
            for term, term_type in zip(triple, ["node", "relation", "node"]):
                term_id = term
                if term_type == "node":
                    if vertex_metadata[term_id]["type"] == self.uid_field:
                        # User
                        prefix = PathLanguageModelingTokenType.USER.token
                        token_str = prefix + str(term_id)
                        token_id = token_vocab[token_str]
                        ret.append(token_id)
                    elif vertex_metadata[term_id]["type"] == self.iid_field:
                        # Item - convert to semantic tokens
                        item_id = term_id - self.user_num
                        if item_id in self.semantic_id_mapping:
                            semantic_ids = self.semantic_id_mapping[item_id]
                            # For graph purposes, use the first semantic token
                            token_str = f"{PathLanguageModelingTokenType.SEMANTIC.token}{semantic_ids[0]}"
                            token_id = token_vocab[token_str]
                            ret.append(token_id)
                        else:
                            # Fallback
                            prefix = PathLanguageModelingTokenType.ITEM.token
                            token_str = prefix + str(item_id)
                            if token_str in token_vocab:
                                token_id = token_vocab[token_str]
                            else:
                                token_id = token_vocab[self.unk_token]
                            ret.append(token_id)
                    elif vertex_metadata[term_id]["type"] == self.entity_field:
                        # Other entity
                        prefix = PathLanguageModelingTokenType.ENTITY.token
                        token_str = prefix + str(term_id - self.user_num)
                        token_id = token_vocab[token_str]
                        ret.append(token_id)
                    else:
                        raise ValueError(
                            f"Unknown vertex type [{vertex_metadata[term_id]['type']}] "
                            "in igraph during tokenized_kg generation."
                        )
                else:
                    # Relation
                    prefix = PathLanguageModelingTokenType.RELATION.token
                    token_str = prefix + str(term_id)
                    token_id = token_vocab[token_str]
                    ret.append(token_id)

            return ret

        tokenized_kg = {}
        for edge in edge_metadata:
            head = edge["source"]
            tail = edge["target"]
            relation = edge["type"]
            relation_id = self.field2token_id[self.relation_field][relation]

            token_ids = igraph_id_to_tokenizer_id(head, relation_id, tail)
            
            # Handle items that map to multiple tokens (we use first token for graph structure)
            head_token = token_ids[0]
            relation_token = token_ids[1]
            tail_token = token_ids[-1]  # Last token if tail is item with multiple semantic tokens

            # Ensure user is always head in user-item relations
            if relation == self.ui_relation and vertex_metadata[head]["type"] != self.uid_field:
                head_token, tail_token = tail_token, head_token

            # Build the graph structure
            if head_token not in tokenized_kg:
                tokenized_kg[head_token] = {}
            if tail_token not in tokenized_kg:
                tokenized_kg[tail_token] = {}

            if relation_token not in tokenized_kg[head_token]:
                tokenized_kg[head_token][relation_token] = set()
            tokenized_kg[head_token][relation_token].add(tail_token)

            if relation_token not in tokenized_kg[tail_token]:
                tokenized_kg[tail_token][relation_token] = set()
            tokenized_kg[tail_token][relation_token].add(head_token)

        return tokenized_kg

    def _get_field_from_config(self):
        super()._get_field_from_config()
        
        # Override token_sequence_length for semantic IDs
        # Worst case: all h entities in path are items → h × semantic_ids_per_item
        max_entity_tokens = self.path_hop_length * self.semantic_ids_per_item
        self.token_sequence_length = (
            1 +                          # User
            self.path_hop_length +       # Relations  
            max_entity_tokens +          # Entities/Items (worst case: all items)
            2                            # BOS/EOS
        )
    
    def get_reverse_semantic_mapping(self):
        """Return mapping: tuple(semantic_ids) → item_id for recommendation extraction."""
        return {
            tuple(sem_ids): item_id 
            for item_id, sem_ids in self.semantic_id_mapping.items()
        }