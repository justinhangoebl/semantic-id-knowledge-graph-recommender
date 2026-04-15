

from typing import Dict, List, Tuple, Set, Optional, Hashable
from hopwise.utils import PathLanguageModelingTokenType
import pandas as pd


class SemanticVocab:
    """Central read-only mapping between item IDs and their semantic token sequences.

    Args:
        semantic_id_mapping: dict[item_id -> list[code_int]]
        tokenizer: tokenizer with `token_to_id(str)` method and SEM{k} tokens
        semantic_ids_per_item: int N (length of every semantic id list)
    """

    def __init__(
        self,
        semantic_id_mapping: Dict[int, List[int]],
        tokenizer,
        semantic_ids_per_item: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._semantic_ids_per_item = int(semantic_ids_per_item)
        self.item_to_token_ids: Dict[int, List[int]] = {}

        # Iterate through each item and its corresponding semantic codes
        for item_id, codes in semantic_id_mapping.items():
            token_ids: List[int] = []
            for code in codes:
                # Use the Enum to get the "SEM" prefix dynamically
                token_name = f"{PathLanguageModelingTokenType.SEMANTIC.token}{int(code)}"
                
                # FIX: Use convert_tokens_to_ids instead of token_to_id
                tid = tokenizer.convert_tokens_to_ids(token_name)
                
                # FIX: PreTrainedTokenizerFast returns int (not None) for unknown tokens,
                # typically returning unk_token_id. Check for that.
                if tid == tokenizer.unk_token_id:
                    # Fail immediately if the tokenizer wasn't built with the codes in the CSV
                    raise ValueError(
                        f"SPRIG Tokenizer mismatch: Token '{token_name}' (required for item {item_id}) "
                        "not found in vocabulary. Ensure _init_tokenizer scanned all possible codes."
                    )
                token_ids.append(int(tid))

            self.item_to_token_ids[int(item_id)] = token_ids

        # Strict collision check to ensure unique semantic mapping
        self.token_ids_to_item: Dict[Tuple[int, ...], int] = {}
        for item_id, tids in self.item_to_token_ids.items():
            key = tuple(tids)
            if key in self.token_ids_to_item:
                raise ValueError(
                    f"Semantic collision: Items {self.token_ids_to_item[key]} and {item_id} share ID {key}."
                )
            self.token_ids_to_item[key] = item_id

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

    def num_semantic_tokens_per_item(self) -> int:
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