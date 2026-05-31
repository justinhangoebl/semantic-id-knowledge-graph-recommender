

import logging
import random

from typing import Dict, List, Tuple, Set, Optional, Hashable
from hopwise.utils import PathLanguageModelingTokenType
import pandas as pd

logger = logging.getLogger(__name__)


class SemanticVocab:
    """Central read-only mapping between item IDs and their semantic token sequences.

    Args:
        semantic_id_mapping: dict[item_id -> list[code_int]]
        tokenizer: tokenizer with `token_to_id(str)` method and SEM{k} tokens
        semantic_ids_per_item: int N (length of every semantic id list)
        resolve_collisions: if True, items whose SEM tuple collides with an earlier
            item get one randomly-chosen position re-drawn until the tuple is unique.
            This preserves semantic structure for non-colliding items (~80%) while
            guaranteeing all items are reachable via the reverse map.
    """

    def __init__(
        self,
        semantic_id_mapping: Dict[int, List[int]],
        tokenizer,
        semantic_ids_per_item: int,
        resolve_collisions: bool = False,
        layered: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._semantic_ids_per_item = int(semantic_ids_per_item)
        self._layered = layered
        self.item_to_token_ids: Dict[int, List[int]] = {}

        sem_prefix = PathLanguageModelingTokenType.SEMANTIC.token

        # Collect all valid SEM token IDs from the tokenizer vocabulary (needed for
        # collision resolution — these are the only codes the model knows about).
        valid_sem_token_ids: List[int] = [
            tid for tok, tid in tokenizer.get_vocab().items()
            if tok.startswith(sem_prefix)
        ]

        for item_id, codes in semantic_id_mapping.items():
            token_ids: List[int] = []
            for level, code in enumerate(codes):
                if layered:
                    token_name = f"SEM_{level}_{int(code)}"
                else:
                    token_name = f"{sem_prefix}{int(code)}"
                tid = tokenizer.convert_tokens_to_ids(token_name)
                if tid == tokenizer.unk_token_id:
                    raise ValueError(
                        f"SPRIG{'L' if layered else ''} Tokenizer mismatch: Token '{token_name}' "
                        f"(required for item {item_id}) not found in vocabulary. "
                        "Ensure _init_tokenizer scanned all possible codes."
                    )
                token_ids.append(int(tid))

            self.item_to_token_ids[int(item_id)] = token_ids

        # Build reverse map; optionally resolve collisions by perturbing one
        # randomly-chosen position of the colliding item until the tuple is unique.
        self.token_ids_to_item: Dict[Tuple[int, ...], int] = {}
        n_collisions = 0
        n_resolved = 0

        for item_id, tids in self.item_to_token_ids.items():
            key = tuple(tids)
            if key not in self.token_ids_to_item:
                self.token_ids_to_item[key] = item_id
                continue

            n_collisions += 1
            if not resolve_collisions:
                # Last-wins: the earlier item keeps the slot, this one is shadowed.
                self.token_ids_to_item[key] = item_id
                continue

            # Resolve: randomly perturb one position at a time until unique.
            resolved = list(tids)
            max_attempts = len(valid_sem_token_ids) * self._semantic_ids_per_item * 10
            for _ in range(max_attempts):
                pos = random.randrange(self._semantic_ids_per_item)
                resolved[pos] = random.choice(valid_sem_token_ids)
                candidate = tuple(resolved)
                if candidate not in self.token_ids_to_item:
                    self.token_ids_to_item[candidate] = item_id
                    self.item_to_token_ids[int(item_id)] = resolved
                    n_resolved += 1
                    break
            else:
                # Extremely unlikely with a large codebook; fall back to last-wins.
                logger.warning(
                    "SemanticVocab: could not resolve collision for item %d after %d attempts; "
                    "item remains shadowed.",
                    item_id, max_attempts,
                )
                self.token_ids_to_item[tuple(tids)] = item_id

        if n_collisions:
            logger.info(
                "SemanticVocab: %d collision(s) detected; %d resolved via perturbation, "
                "%d shadowed (resolve_collisions=%s).",
                n_collisions, n_resolved, n_collisions - n_resolved, resolve_collisions,
            )

    # Public API
    def get_item_tokens(self, item_id: int) -> List[int]:
        
        return list(self.item_to_token_ids[int(item_id)])

    def get_first_token(self, item_id: int) -> int:
        return int(self.item_to_token_ids[int(item_id)][0])

    def reverse_map(self, token_ids: Tuple[int, ...]) -> Optional[int]:
        key = tuple(int(x) for x in token_ids)
        return self.token_ids_to_item.get(key)

    def all_item_ids(self) -> Set[int]:
        return set(self.item_to_token_ids.keys())

    def num_semantic_tokens(self) -> int:
        return int(self._semantic_ids_per_item)

    # convenience dunders
    def __len__(self) -> int:
        return len(self.item_to_token_ids)

    def __contains__(self, item_id: int) -> bool:
        return int(item_id) in self.item_to_token_ids


    @staticmethod
    def load_semantic_ids(file_name: str, delim: str="\t", column_name: str="semantic_ids") -> dict[int, list[int]]:
        from typing import cast
        return cast(
            dict[int, list[int]],
            pd.read_csv(file_name, sep=delim, index_col=0)[column_name]
            .str.strip("[]")
            .str.split(", ")
            .apply(lambda x: [int(i) for i in x])
            .to_dict()
        )