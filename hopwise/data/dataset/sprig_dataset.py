"""
SPRIG Dataset — semantic-ID-based knowledge path dataset.

Items are represented as N consecutive SEM tokens instead of a single I token.
No fallbacks: if semantic IDs are not present or malformed, the dataset raises.
"""

import os
import pickle

import joblib
import numpy as np

from hopwise.data import Interaction
from hopwise.data.dataset import KnowledgePathDataset
from hopwise.data.dataset.kgglm_dataset import _generate_paths_random_walks
from hopwise.data.semantic_vocab import SemanticVocab
from hopwise.utils import PathLanguageModelingTokenType, progress_bar, set_color


class SPRIGDataset(KnowledgePathDataset):
    """KnowledgePathDataset variant where items are encoded as N SEM tokens.

    The parent tokenizer (with I-tokens) is replaced entirely by a SEM-token
    tokenizer built in ``_init_tokenizer``.  Every other dataset mechanism
    (KG loading, path sampling, relation interleaving) is inherited unchanged.

    Required config keys (on top of parent requirements)
    -------------------------------------------------------
    train_stage           : "pretrain" | "finetune"
    semantic_ids_per_item : int  — N, number of SEM codes per item
    semantic_ids_file     : str  — path to the .semanticids CSV
    path_sample_args.pretrain_hop_length : "(min, max)"
    path_sample_args.pretrain_paths      : int
    """
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str
    mask_token: str
    path_token_separator: str
    special_tokens: list[str]
    context_length: int
    tokenizer_model: str

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config):
        # _get_field_from_config() is called by the grandparent before this
        # body runs, so _raw_semantic_mapping is already populated.
        super().__init__(config)  # also calls _init_tokenizer → builds semantic_vocab

        # Hard check: I-tokens must not exist in the SPRIG tokenizer.
        unk_id = self._tokenizer.unk_token_id
        item_tok = PathLanguageModelingTokenType.ITEM.token + "0"
        assert self._tokenizer.convert_tokens_to_ids(item_tok) == unk_id, (
            "SPRIG tokenizer must contain no ITEM tokens. "
            f"Found '{item_tok}' mapped to a real id — check _init_tokenizer."
        )

        self.logger.info(
            "SPRIGDataset ready: train_stage=%s, N=%d, K=%d SEM types, "
            "%d items with semantic IDs.",
            self.train_stage,
            self.semantic_ids_per_item,
            self.semantic_vocab.num_semantic_tokens_per_item(),
            len(self._raw_semantic_mapping),
        )

    # ------------------------------------------------------------------
    # Config loading (runs before __init__ body via grandparent)
    # ------------------------------------------------------------------

    def _get_field_from_config(self):
        super()._get_field_from_config()

        # --- Required SPRIG keys — KeyError if absent, intentional. ---
        self.train_stage = self.config["train_stage"]
        self.semantic_ids_per_item = int(self.config["semantic_ids_per_item"])
        
        path_args = self.config["path_sample_args"]
        pretrain_hop = path_args["pretrain_hop_length"]
        if isinstance(pretrain_hop, str):
            pretrain_hop = tuple(map(int, pretrain_hop.strip("()[] ").split(",")))
        self.pretrain_hop_length: tuple[int, int] = tuple(pretrain_hop)  # type: ignore
        self.pretrain_paths = int(path_args["pretrain_paths"])


        # Each stage uses its own exact sequence length so padding and n_positions
        # are never over-allocated.  A mismatch between pretrain and finetune
        # n_positions is resolved by the WPE adaptation in sprig.py (intentional).
        # [BOS] U/every [R every] * hop + [EOS]
        if self.train_stage == "finetune":
            self.token_sequence_length = 2 + 1 + self.path_hop_length * (self.semantic_ids_per_item + 1)
        else:
            # Pretrain paths include two additional boundary terms compared to
            # the previous closed-form approximation.
            self.token_sequence_length = (
                2
                + self.semantic_ids_per_item
                + self.pretrain_hop_length[1] * (self.semantic_ids_per_item + 1)
                + 2
            )

        # SPRIG uses an exact stage-specific sequence length.
        # Keep tokenizer padding/truncation length aligned with model n_positions
        # to avoid positional embedding overflows.
        if self.context_length != self.token_sequence_length:
            self.logger.info(
                "SPRIG: overriding context_length from %d to exact token_sequence_length %d.",
                self.context_length,
                self.token_sequence_length,
            )
        self.context_length = self.token_sequence_length

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        sem_file = os.path.join(self.dataset_path, self.config["dataset"] + ".semanticids")
        self._raw_semantic_mapping = self._load_semantic_id_mapping(sem_file)


    # ------------------------------------------------------------------
    # Semantic ID file loading
    # ------------------------------------------------------------------

    def _load_semantic_id_mapping(self, filepath: str) -> dict[int, list[int]]:
        """Load ``filepath`` → ``{item_id: [code_0, …, code_{N-1}]}``.

        Raises on every error condition — no silent fallbacks.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Semantic IDs file not found: {filepath}\n"
                "Set 'semantic_ids_file' in your config to a valid path."
            )

        mapping = SemanticVocab.load_semantic_ids(
            file_name=filepath, delim=","
        )
        if not mapping:
            raise ValueError(f"Semantic IDs file is empty: {filepath}")

        N = self.semantic_ids_per_item

        # Every item must have exactly N codes.
        bad = [iid for iid, codes in mapping.items() if len(codes) != N]
        if bad:
            raise ValueError(
                f"Items with wrong code count (expected {N}): {bad[:10]}"
                + (" …" if len(bad) > 10 else "")
            )

        # No two items may share the same code tuple (would break reverse map).
        seen: dict[tuple, int] = {}
        duplicate = []
        for iid, codes in mapping.items():
            key = tuple(codes)
            if key in seen:
                duplicate.append((seen[key], iid))
            seen[key] = iid
        
        if len(duplicate) > 0:
            raise ValueError(duplicate, len(duplicate))
        return mapping

    # ------------------------------------------------------------------
    # Tokenizer (replaces parent's I-token version)
    # ------------------------------------------------------------------

    def _init_tokenizer(self):
        from tokenizers import Tokenizer, pre_tokenizers
        from tokenizers import models as token_models
        from tokenizers import processors as token_processors
        from tokenizers import trainers as token_trainers
        from transformers import PreTrainedTokenizerFast

        # K = number of distinct SEM token types (max observed code + 1).
        all_codes = [c for codes in self._raw_semantic_mapping.values() for c in codes]
        K = int(max(all_codes)) + 1

        # entity_range: non-item entity IDs (item_num … entity_num-1).
        entity_range = np.arange(self.item_num, self.entity_num)

        token_vocab = np.concatenate([
            np.char.add(PathLanguageModelingTokenType.USER.token,
                        np.arange(self.user_num).astype(str)),
            np.char.add(PathLanguageModelingTokenType.SEMANTIC.token,
                        np.arange(K).astype(str)),          # SEM0 … SEM{K-1}
            np.char.add(PathLanguageModelingTokenType.ENTITY.token,
                        entity_range.astype(str)),
            np.char.add(PathLanguageModelingTokenType.RELATION.token,
                        np.arange(self.relation_num).astype(str)),
        ])

        tokenizer_model_cls = getattr(token_models, self.tokenizer_model)
        tok_obj = Tokenizer(tokenizer_model_cls(unk_token=self.unk_token))
        tok_obj.pre_tokenizer = pre_tokenizers.Split(
            self.path_token_separator, "removed"
        ) # type: ignore

        trainer_cls = getattr(token_trainers, self.tokenizer_model + "Trainer")
        trainer = trainer_cls(
            vocab_size=len(token_vocab) + len(self.special_tokens),
            special_tokens=self.special_tokens,
        )
        tok_obj.train_from_iterator(token_vocab, trainer=trainer)

        from tokenizers.processors import TemplateProcessing

        # TemplateProcessing requires both single and pair sequences
        tok_obj.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.eos_token} $B:1 {self.eos_token}:1",  # Add this
            special_tokens=[
                (self.bos_token, tok_obj.token_to_id(self.bos_token)),
                (self.eos_token, tok_obj.token_to_id(self.eos_token)),
            ],
        ) # type: ignore

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok_obj,
            model_max_length=self.context_length,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token,
        )

        # SemanticVocab is built once here; every component that needs it
        # gets it through dataset.semantic_vocab.
        self.semantic_vocab = SemanticVocab(
            self._raw_semantic_mapping,
            self._tokenizer,
            self.semantic_ids_per_item,
        )

    # ------------------------------------------------------------------
    # Path formatting — items → N SEM tokens
    # ------------------------------------------------------------------
    def _format_path(self, path: np.ndarray) -> str | None:
        """Convert a raw relation-interleaved path array to a token string.

        Item nodes are expanded to N space-separated SEM tokens.
        Returns ``None`` if any item in the path lacks a semantic mapping —
        the caller must skip such paths.
        """
        path = path[path != self.PATH_PADDING]
        path_nodes = path[::2]       # positions 0, 2, 4 … → node vertex IDs
        path_relations = path[1::2]  # positions 1, 3, 5 … → relation IDs

        graph_min_iid = self.user_num
        graph_max_iid = self.user_num + self.item_num - 1

        # Build per-node token lists.
        node_token_lists: list[list[str]] = []
        for node in path_nodes:
            if isinstance(node, str):
                prefix_length = PathLanguageModelingTokenType.get_prefix_length(node)
                n = int(node[prefix_length:])
            else:
                n = int(node)
            if n < graph_min_iid:
                # User vertex
                node_token_lists.append(
                    [PathLanguageModelingTokenType.USER.token + str(n)]
                )
            elif n <= graph_max_iid:
                # Item vertex — expand to N SEM tokens.
                item_id = n - self.user_num
                if item_id not in self._raw_semantic_mapping:
                    return None  # path contains unmapped item — drop it
                codes = self._raw_semantic_mapping[item_id]
                node_token_lists.append(
                    [PathLanguageModelingTokenType.SEMANTIC.token + str(c) for c in codes]
                )
            else:
                # Non-item entity vertex.
                # entity_range starts at item_num, so the token is E{n - user_num}.
                node_token_lists.append(
                    [PathLanguageModelingTokenType.ENTITY.token + str(n - self.user_num)]
                )

        relation_tokens = [
            PathLanguageModelingTokenType.RELATION.token + str(int(r))
            for r in path_relations
        ]

        # Interleave: node0 rel0 node1 rel1 …
        out: list[str] = list(node_token_lists[0])
        for rel_tok, node_toks in zip(relation_tokens, node_token_lists[1:]):
            out.append(rel_tok)
            out.extend(node_toks)

        return self.path_token_separator.join(out)

    # ------------------------------------------------------------------
    # Path dataset generation
    # ------------------------------------------------------------------

    def generate_user_path_dataset(self):
        """Entry point for both pretrain and finetune path generation."""
        if self.train_stage == "pretrain":
            self.generate_pretrain_dataset()
            return

        if self._path_dataset is not None:
            return

        generated_paths = self.generate_user_paths()
        lines: list[str] = []
        skipped = 0
        for path in generated_paths:
            fmt = self._format_path(path)
            if fmt is not None:
                lines.append(fmt)
            else:
                skipped += 1

        self.logger.info(
            "SPRIG finetune: %d paths formatted, %d skipped (missing semantic mapping).",
            len(lines),
            skipped,
        )
        self._path_dataset = "\n".join(lines)

    def generate_pretrain_dataset(self):
        """Generate (or load from cache) pretrain paths and format them."""
        if self._path_dataset is not None:
            return

        wandb_proj = getattr(self.config, "wandb_project", "SPRIG_default")
        cache_file = (
            f"./paths/{wandb_proj}_sprig_pretrain_raw_{self.pretrain_hop_length}.pkl"
        )

        if os.path.exists(cache_file) and self.config["use_cached_pretrain_paths"]:
            self.logger.info(
                "Loading cached SPRIG pretrain formatted paths from %s …", cache_file
            )
            with open(cache_file, "rb") as fh:
                cached_paths = pickle.load(fh)

            # Backward compatibility:
            # - new cache: list[str] of already-formatted paths
            # - old cache: ndarray/list of raw relation-interleaved arrays
            if isinstance(cached_paths, (list, tuple)) and (
                len(cached_paths) == 0 or isinstance(cached_paths[0], str)
            ):
                self._path_dataset = "\n".join(cached_paths)
                return

            self.logger.warning(
                "SPRIG cache contains legacy raw path format. Re-formatting and updating cache: %s",
                cache_file,
            )
            raw_paths = cached_paths
        else:
            # Sample new paths from KG and format them.
            self.logger.info(cache_file)
            raw_paths = self._sample_pretrain_paths()  # returns numpy array of path arrays

        lines: list[str] = []
        skipped = 0
        for path in raw_paths:
            fmt = self._format_path(path)
            if fmt is not None:
                lines.append(fmt)
            else:
                skipped += 1

        self.logger.info(
            "SPRIG pretrain: %d paths formatted, %d skipped.", len(lines), skipped
        )

        # Cache the formatted strings for next time
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        with open(cache_file, "wb") as fh:
            pickle.dump(lines, fh)
        self.logger.info(
            "Cached %d pretrain formatted paths to %s.", len(lines), cache_file
        )

        self._path_dataset = "\n".join(lines)

    def tokenize_path_dataset(self):
        """Tokenize pre-generated paths and keep only structurally valid rows.

        Unlike the parent implementation, PAD tokens are allowed because pretrain
        paths can be shorter than `context_length` and are padded by the tokenizer.
        """
        if self._tokenized_dataset is not None:
            return

        if self._tokenizer is None:
            raise ValueError("Tokenizer has not been initialized.")

        tokenized_dataset = self.tokenize(self.path_dataset.split("\n"))
        tokenized_dataset = Interaction(tokenized_dataset.data)

        def _token_id(tok: str) -> int:
            tok_id = self._tokenizer.convert_tokens_to_ids(tok)
            if not isinstance(tok_id, int):
                raise ValueError(f"Unexpected token id type for token '{tok}': {type(tok_id)}")
            return tok_id

        unk_id = _token_id(self.unk_token)
        pad_id = _token_id(self.pad_token)
        bos_id = _token_id(self.bos_token)
        eos_id = _token_id(self.eos_token)

        allowed_inside = {pad_id}
        disallowed_inside = set(self._tokenizer.all_special_ids) - allowed_inside

        valid_mask = []
        for path_ids in tokenized_dataset["input_ids"]:
            ids = path_ids.tolist()

            # Must start/end with BOS/EOS and contain no UNK.
            if (
                not ids
                or ids[0] != bos_id
                or eos_id not in ids
                or unk_id in ids
            ):
                valid_mask.append(False)
                continue

            # Ignore tail padding after EOS when checking internal special tokens.
            eos_pos = len(ids) - 1 - ids[::-1].index(eos_id)
            inside = ids[1:eos_pos]
            valid = all(tok not in disallowed_inside for tok in inside)
            valid_mask.append(valid)

        kept = int(sum(valid_mask))
        total = len(tokenized_dataset)
        if kept == 0 and total > 0:
            self.logger.warning(
                "SPRIG tokenization filter removed all paths (%d/%d). "
                "Falling back to unfiltered tokenized dataset to avoid empty training set.",
                total,
                total,
            )
            self._tokenized_dataset = tokenized_dataset
        else:
            self._tokenized_dataset = tokenized_dataset[valid_mask]

        self.logger.info(
            "SPRIG tokenization: kept %d/%d paths after validation.",
            len(self._tokenized_dataset),
            total,
        )

    def _sample_pretrain_paths(self) -> np.ndarray:
        """Run random walks on the KG and return relation-interleaved path arrays."""
        graph = self._create_ckg_igraph(show_relation=True, directed=False)
        kg_rel_num = len(self.relations)
        graph.es["weight"] = [0.0] * self.inter_num + [1.0] * kg_rel_num

        # Strip zero-weight interaction edges for sampling — they are never
        # traversed but inflate average vertex degree ~10x, making every
        # random_walk step scan far more neighbors than needed.
        # The full graph is kept for _add_paths_relations_variable lookups.
        kg_only_graph = graph.copy()
        kg_only_graph.delete_edges(kg_only_graph.es.select(weight_eq=0))

        min_hop, max_hop = self.pretrain_hop_length
        graph_min_iid = self.user_num

        paths: set[tuple] = set()
        max_tries = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]
        iter_entities = progress_bar(
            range(graph_min_iid, len(kg_only_graph.vs)),
            desc=set_color("SPRIG Pretrain Sampling", "red", progress=True),
            ncols=100,
        )

        kwargs = dict(
            graph=kg_only_graph,
            min_hop=min_hop,
            max_hop=max_hop,
            pretrain_paths=self.pretrain_paths,
            max_tries_per_entity=max_tries,
            paths=paths,
        )

        if not self.parallel_max_workers:
            for entity in iter_entities:
                _generate_paths_random_walks(entity, **kwargs)
        else:
            list(
                joblib.Parallel(n_jobs=self.parallel_max_workers, prefer="threads")(
                    joblib.delayed(_generate_paths_random_walks)(entity, **kwargs) for entity in iter_entities
                )
            )

        return self._add_paths_relations_variable(graph, paths, max_hop)

    def _add_paths_relations_variable(
        self, graph, paths: set, max_hop: int
    ) -> np.ndarray:
        """Interleave relation IDs into vertex-only paths of variable length.

        The parent's ``_add_paths_relations`` pre-allocates based on the fixed
        ``self.path_hop_length``, which is wrong for pretrain paths that can be
        shorter or longer.  This version accepts an explicit ``max_hop``.

        Returns a 2-D numpy array of shape (n_paths, 2*max_hop+1) with
        ``PATH_PADDING`` filling unused positions.
        """
        # Build a sparse edge → relation lookup from the igraph edge list.
        # Keys are (source_vid, target_vid) tuples; values are relation token IDs.
        rel_field = self.field2token_id[self.relation_field]
        edge_rel: dict[tuple[int, int], int] = {}
        for e in graph.es:
            rid = rel_field[e["type"]]
            edge_rel[(e.source, e.target)] = rid
            if not graph.is_directed():
                edge_rel[(e.target, e.source)] = rid

        complete_len = 2 * max_hop + 1
        n = len(paths)
        result = np.full((n, complete_len), fill_value=self.PATH_PADDING, dtype=int)

        for i, path in enumerate(paths):
            nodes = list(path)
            pos = 0
            for j, node in enumerate(nodes):
                if pos >= complete_len:
                    break
                result[i, pos] = node
                pos += 1
                if j + 1 < len(nodes) and pos < complete_len:
                    result[i, pos] = edge_rel.get(
                        (int(node), int(nodes[j + 1])), self.PATH_PADDING
                    )
                    pos += 1

        return result

    # ------------------------------------------------------------------
    # CKG and used-ID helpers (required by the framework)
    # ------------------------------------------------------------------

    def get_tokenized_ckg(self) -> dict:
        """Return the CKG as ``head_tok → rel_tok → set(tail_tok_tuple)``.

        Item tails are N-tuples of SEM token IDs.
        Non-item nodes (users, entities) are 1-tuples for type consistency.
        """
        vocab = self._tokenizer.get_vocab()
        graph = self._create_ckg_igraph(show_relation=True, directed=False)
        rel_field = self.field2token_id[self.relation_field]

        graph_min_iid = self.user_num
        graph_max_iid = self.user_num + self.item_num - 1
        semantic_items = self.semantic_vocab.all_item_ids()

        def vertex_to_tok_tuple(v: int) -> tuple[int, ...] | None:
            if v < graph_min_iid:
                return (vocab[PathLanguageModelingTokenType.USER.token + str(v)],)
            elif v <= graph_max_iid:
                item_id = v - self.user_num
                if item_id not in semantic_items:
                    return None
                return tuple(self.semantic_vocab.get_item_tokens(item_id))
            else:
                return (vocab[PathLanguageModelingTokenType.ENTITY.token + str(v - self.user_num)],)

        tokenized_kg: dict = {}
        skipped_edges = 0
        for e in graph.es:
            head_tup = vertex_to_tok_tuple(e.source)
            tail_tup = vertex_to_tok_tuple(e.target)
            if head_tup is None or tail_tup is None:
                skipped_edges += 1
                continue

            rel_id = rel_field[e["type"]]
            rel_tok = vocab[PathLanguageModelingTokenType.RELATION.token + str(rel_id)]

            # Add both directions (graph is undirected).
            for h, t in ((head_tup, tail_tup), (tail_tup, head_tup)):
                tokenized_kg.setdefault(h, {}).setdefault(rel_tok, set()).add(t)

        if skipped_edges > 0:
            self.logger.warning(
                "SPRIG tokenized CKG: skipped %d edges due to missing semantic mapping.",
                skipped_edges,
            )

        return tokenized_kg

    def get_tokenized_used_ids(self) -> dict[int, set[tuple[int, ...]]]:
        """Return ``user_token_id → set of SEM-tuple item representations``."""
        used_ids = self.get_user_used_ids()
        vocab = self._tokenizer.get_vocab()

        result: dict[int, set[tuple[int, ...]]] = {}
        items_iter = (
            used_ids.items() if isinstance(used_ids, dict) else enumerate(used_ids)
        )
        skipped_items = 0
        for uid, items in items_iter:
            u_tok = vocab[PathLanguageModelingTokenType.USER.token + str(uid)]
            user_items: set[tuple[int, ...]] = set()
            for iid in items:
                item_id = int(iid)
                if item_id not in self.semantic_vocab:
                    skipped_items += 1
                    continue
                user_items.add(tuple(self.semantic_vocab.get_item_tokens(item_id)))
            result[u_tok] = user_items

        if skipped_items > 0:
            self.logger.warning(
                "SPRIG tokenized used IDs: skipped %d items due to missing semantic mapping.",
                skipped_items,
            )

        return result