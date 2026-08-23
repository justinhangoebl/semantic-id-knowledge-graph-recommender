"""
SPRIG Dataset — semantic-ID-based knowledge path dataset.

Items are represented as N consecutive SEM tokens instead of a single I token.
No fallbacks: if semantic IDs are not present or malformed, the dataset raises.
"""

import os
import pickle

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
            "%d / %d filtered items covered by semantic IDs.",
            self.train_stage,
            self.semantic_ids_per_item,
            self.semantic_vocab.num_semantic_tokens(),
            len(self.semantic_vocab),
            self.item_num - 1,
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


        self.layered_semantic_ids = bool(
            self.config["layered_semantic_ids"] if "layered_semantic_ids" in self.config else False
        )
        self.resolve_semantic_id_collisions = bool(
            self.config["resolve_semantic_id_collisions"]
            if "resolve_semantic_id_collisions" in self.config
            else False
        )

        if self.train_stage == "finetune":
            # Exact length: BOS + U + H*(R + 1 intermediate token) + 2*N SEM tokens + EOS
            # = 2*H + 2*N + 1  (intermediate nodes are always 1 token — entity or user)
            self.token_sequence_length = 2 * self.path_hop_length + 2 * self.semantic_ids_per_item + 1
        else:
            # Worst case: every node in the walk is an item (N SEM tokens each).
            # h+1 nodes, h relations, +2 for BOS/EOS, maximized at h = max_hop.
            max_hop = self.pretrain_hop_length[1]
            self.token_sequence_length = (max_hop + 1) * self.semantic_ids_per_item + max_hop + 2

        # Always clamp context_length to the exact computed length so the tokenizer
        # does not pad paths shorter than context_length.  Padding tokens inside
        # path[1:-1] cause tokenize_path_dataset to discard every path.
        if self.context_length != self.token_sequence_length:
            self.logger.info(
                "SPRIG: setting context_length to exact token_sequence_length %d (was %d).",
                self.token_sequence_length,
                self.context_length,
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
        """Load ``filepath`` ... ``{item_id: [code_0, …, code_{N-1}]}``."""
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

        # Check for items sharing the same code tuple (collision in the reverse map).
        # Collisions mean the second item in a pair becomes unreachable during generation;
        # we warn and continue with last-wins semantics rather than aborting.
        seen: dict[tuple, int] = {}
        duplicate = []
        for iid, codes in mapping.items():
            key = tuple(codes)
            if key in seen:
                duplicate.append((seen[key], iid))
            seen[key] = iid

        if duplicate:
            if self.resolve_semantic_id_collisions:
                self.logger.warning(
                    "SPRIG: %d semantic ID collision(s) found in %s. "
                    "resolve_semantic_id_collisions=True: colliding items will get one "
                    "code position re-drawn until unique.",
                    len(duplicate),
                    filepath,
                )
            else:
                self.logger.warning(
                    "SPRIG: %d semantic ID collision(s) found in %s. "
                    "Colliding items are unreachable during generation (last-wins). "
                    "Re-run semantic ID assignment, or set resolve_semantic_id_collisions: True, "
                    "to eliminate collisions.",
                    len(duplicate),
                    filepath,
                )
        return mapping

    # ------------------------------------------------------------------
    # Tokenizer (replaces parent's I-token version)
    # ------------------------------------------------------------------

    def _extra_vocab_segments(self) -> list[np.ndarray]:
        """Additional token-array segments appended to the vocab after USER/SEM/ENTITY.

        Overridable so relation-free variants (e.g. SPRIGT) can drop the
        RELATION token block entirely without duplicating the rest of
        ``_init_tokenizer``.
        """
        return [
            np.char.add(PathLanguageModelingTokenType.RELATION.token,
                        np.arange(self.relation_num).astype(str)),
        ]

    def _init_tokenizer(self):
        from tokenizers import Tokenizer, pre_tokenizers
        from tokenizers import models as token_models
        from tokenizers import processors as token_processors
        from tokenizers import trainers as token_trainers
        from transformers import PreTrainedTokenizerFast

        # entity_range: non-item entity IDs (item_num … entity_num-1).
        entity_range = np.arange(self.item_num, self.entity_num)

        if self.layered_semantic_ids:
            # Layered vocab: SEM_{l}_{k} per hierarchy level l, code k.
            # Compute per-layer code range from the data.
            N = self.semantic_ids_per_item
            per_layer_codes: list[list[int]] = [[] for _ in range(N)]
            for codes in self._raw_semantic_mapping.values():
                for l, c in enumerate(codes):
                    per_layer_codes[l].append(c)
            K_per_layer = [int(max(pc)) + 1 for pc in per_layer_codes]
            sem_tokens_list = [
                f"SEM_{l}_{k}"
                for l in range(N)
                for k in range(K_per_layer[l])
            ]
            sem_token_array = np.array(sem_tokens_list)
        else:
            # Flat vocab: SEM{k} shared across all levels.
            all_codes = [c for codes in self._raw_semantic_mapping.values() for c in codes]
            K = int(max(all_codes)) + 1
            sem_token_array = np.char.add(PathLanguageModelingTokenType.SEMANTIC.token,
                                          np.arange(K).astype(str))

        token_vocab = np.concatenate([
            np.char.add(PathLanguageModelingTokenType.USER.token,
                        np.arange(self.user_num).astype(str)),
            sem_token_array,
            np.char.add(PathLanguageModelingTokenType.ENTITY.token,
                        entity_range.astype(str)),
            *self._extra_vocab_segments(),
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

        # Translate raw mapping (keyed by original item token, e.g. movie ID "1193")
        # to internal RecBole item indices so all downstream lookups use a consistent
        # ID space.  field2token_id[iid_field] maps original token string → internal idx.
        item_token2id: dict = self.field2token_id[self.iid_field]
        self._internal_semantic_mapping: dict[int, list[int]] = {}
        for orig_id, codes in self._raw_semantic_mapping.items():
            internal_id = item_token2id.get(str(orig_id))
            if internal_id is not None:
                self._internal_semantic_mapping[int(internal_id)] = codes

        n_covered = len(self._internal_semantic_mapping)
        n_items = self.item_num - 1  # item_num includes [PAD] at index 0
        self.logger.info(
            "SPRIG semantic ID coverage: %d / %d filtered items have semantic IDs "
            "(%d items in file had no match after filtering).",
            n_covered, n_items, n_items - n_covered,
        )

        # SemanticVocab is built once here; every component that needs it
        # gets it through dataset.semantic_vocab.
        self.semantic_vocab = SemanticVocab(
            self._internal_semantic_mapping,
            self._tokenizer,
            self.semantic_ids_per_item,
            resolve_collisions=self.resolve_semantic_id_collisions,
            layered=self.layered_semantic_ids,
        )

    # ------------------------------------------------------------------
    # Path formatting — items → N SEM tokens
    # ------------------------------------------------------------------

    def _path_node_token_lists(self, path_nodes) -> list[list[str]] | None:
        """Map each raw vertex ID in ``path_nodes`` to its token list.

        User/entity vertices map to a single token; item vertices expand to
        N SEM tokens. Returns ``None`` if any item in the path lacks a
        semantic mapping — the caller must skip such paths.
        """
        graph_min_iid = self.user_num
        graph_max_iid = self.user_num + self.item_num - 1

        node_token_lists: list[list[str]] = []
        for node in path_nodes:
            n = int(node)
            if n < graph_min_iid:
                # User vertex
                node_token_lists.append(
                    [PathLanguageModelingTokenType.USER.token + str(n)]
                )
            elif n <= graph_max_iid:
                # Item vertex — expand to N SEM tokens.
                item_id = n - self.user_num  # internal RecBole index
                if item_id not in self._internal_semantic_mapping:
                    return None  # path contains unmapped item — drop it
                codes = self._internal_semantic_mapping[item_id]
                if self.layered_semantic_ids:
                    node_token_lists.append(
                        [f"SEM_{l}_{c}" for l, c in enumerate(codes)]
                    )
                else:
                    node_token_lists.append(
                        [PathLanguageModelingTokenType.SEMANTIC.token + str(c) for c in codes]
                    )
            else:
                # Non-item entity vertex.
                # entity_range starts at item_num, so the token is E{n - user_num}.
                node_token_lists.append(
                    [PathLanguageModelingTokenType.ENTITY.token + str(n - self.user_num)]
                )
        return node_token_lists

    def _format_path(self, path: np.ndarray) -> str | None:
        """Convert a raw relation-interleaved path array to a token string.

        Item nodes are expanded to N space-separated SEM tokens.
        Returns ``None`` if any item in the path lacks a semantic mapping —
        the caller must skip such paths.
        """
        path = path[path != self.PATH_PADDING]
        path_nodes = path[::2]       # positions 0, 2, 4 … → node vertex IDs
        path_relations = path[1::2]  # positions 1, 3, 5 … → relation IDs

        node_token_lists = self._path_node_token_lists(path_nodes)
        if node_token_lists is None:
            return None

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
        raw_paths = self._sample_pretrain_paths()

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
        self._path_dataset = "\n".join(lines)

    def _sample_pretrain_paths(self) -> np.ndarray:
        """Run random walks on the KG and return relation-interleaved path arrays."""
        graph = self._create_ckg_igraph(show_relation=True, directed=False)
        kg_rel_num = len(self.relations)
        # Weight 0 on user-item edges keeps walks in the KG sub-graph.
        graph.es["weight"] = [0.0] * self.inter_num + [1.0] * kg_rel_num

        min_hop, max_hop = self.pretrain_hop_length
        graph_min_iid = self.user_num + 1  # skip user vertices

        paths: set[tuple] = set()
        max_tries = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]

        for entity in progress_bar(
            range(graph_min_iid, len(graph.vs)),
            desc=set_color("SPRIG Pretrain Sampling", "red", progress=True),
            ncols=100,
        ):
            _generate_paths_random_walks(
                entity,
                graph=graph,
                min_hop=min_hop,
                max_hop=max_hop,
                pretrain_paths=self.pretrain_paths,
                max_tries_per_entity=max_tries,
                paths=paths,
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
    # Tokenization (overrides parent — SPRIG paths have variable length)
    # ------------------------------------------------------------------

    def tokenize_path_dataset(self):
        """Tokenize the path dataset, tolerating trailing padding.

        Unlike KGGLM's fixed-width I-tokens, SPRIG items expand to
        ``semantic_ids_per_item`` tokens while non-item nodes stay at 1
        token, so paths have variable real length and most get right-padded
        below ``context_length``. The parent's mask (``KnowledgePathDataset.
        tokenize_path_dataset``) rejects any row containing a special token
        in ``path[1:-1]``, which — for right-padded rows — always includes
        the trailing [PAD]/EOS shift and discards the row. That silently
        drops the majority of SPRIG pretrain data. This override checks
        only the real (non-padded) interior of each row, via attention_mask.
        """
        if self._tokenized_dataset is None:
            path_rows = [path for path in self.path_dataset.splitlines() if path.strip()]
            if not path_rows:
                raise ValueError(
                    "Path dataset is empty — no valid paths were generated. "
                    "Check path_sample_args (strategy, restrict_by_phase, temporal_causality) "
                    "and that the KG has entity connections between items."
                )
            tokenized_dataset = self.tokenize(path_rows)
            tokenized_dataset = Interaction(tokenized_dataset.data)
            special_ids = set(self.tokenizer.all_special_ids)
            correct_path_mask = [
                all(
                    int(tok) not in special_ids
                    for tok in path[1 : int(attn_mask.sum()) - 1]
                )
                for path, attn_mask in zip(
                    tokenized_dataset["input_ids"], tokenized_dataset["attention_mask"]
                )
            ]
            tokenized_dataset = tokenized_dataset[correct_path_mask]
            self._tokenized_dataset = tokenized_dataset

    # ------------------------------------------------------------------
    # CKG and used-ID helpers (required by the framework)
    # ------------------------------------------------------------------

    def _edge_relation_token(self, rel_id: int, vocab: dict) -> int:
        """Map a raw relation id to the token id used to key ``get_tokenized_ckg``'s
        inner dict. Overridable so relation-free variants (e.g. SPRIGT), whose
        vocab has no RELATION tokens, can supply a placeholder instead of a
        vocab lookup that would KeyError.
        """
        return vocab[PathLanguageModelingTokenType.RELATION.token + str(rel_id)]

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

        def vertex_to_tok_tuple(v: int) -> tuple[int, ...] | None:
            if v < graph_min_iid:
                return (vocab[PathLanguageModelingTokenType.USER.token + str(v)],)
            elif v <= graph_max_iid:
                item_id = v - self.user_num
                try:
                    return tuple(self.semantic_vocab.get_item_tokens(item_id))
                except KeyError:
                    return None
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
            rel_tok = self._edge_relation_token(rel_id, vocab)

            # Add both directions (graph is undirected).
            for h, t in ((head_tup, tail_tup), (tail_tup, head_tup)):
                tokenized_kg.setdefault(h, {}).setdefault(rel_tok, set()).add(t)

        if skipped_edges > 0:
            self.logger.warning(
                "SPRIG tokenized CKG: skipped %d edges — item missing from semantic vocab "
                "(dataset filter broader than vocab coverage).",
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
        for uid, items in items_iter:
            u_tok = vocab[PathLanguageModelingTokenType.USER.token + str(uid)]
            result[u_tok] = {
                tuple(self.semantic_vocab.get_item_tokens(int(iid)))
                for iid in items
                if int(iid) in self.semantic_vocab
            }
        return result