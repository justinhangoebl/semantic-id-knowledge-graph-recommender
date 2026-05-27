# SPRIG: Semantic Path Reasoning with Interpretable Generation

This repository contains the implementation of **SPRIG**, a knowledge-graph-based recommender system that replaces raw item identifiers with learned semantic codes, enabling compositional generalization over items through shared representation structure. The codebase is a fork of [hopwise](https://github.com/tail-unica/hopwise), an open-source library for explainable path-reasoning recommendation over knowledge graphs.

---

## Overview

Existing path language modeling methods for recommendation — such as PEARLM and KGGLM — represent items as opaque atomic tokens. While this allows an autoregressive model to generate reasoning paths through a knowledge graph, it treats each item as an isolated symbol. The model must learn item-specific predictions entirely from co-occurrence patterns, with no inductive bias toward items that share semantic properties.

SPRIG addresses this by replacing item tokens with **semantic ID tuples**: short sequences of discrete codes derived from a hierarchical quantization of item embeddings (e.g., residual quantization or k-means over entity representations). An item `i` is no longer a single token `I_i` but a sequence `SEM_{c_1} SEM_{c_2} ... SEM_{c_N}`. Items that are semantically similar share code prefixes, so the model can leverage partial matches during generation — a form of compositional reasoning that atomic tokens do not afford.

The path generation task is otherwise unchanged. The model takes a user token as a prefix and autoregressively generates a reasoning path through the knowledge graph, where item positions in the path are expanded to their N-token semantic representations. Recommendation candidates are extracted from the generated paths by reading off the terminal item's code tuple and looking it up in the reverse semantic vocabulary.

---

## Method

### Semantic Identifiers

Each item in the dataset is assigned a tuple of integer codes $(c_1, c_2, \ldots, c_N)$ from a shared codebook of size $K$. The codes are derived offline from item embeddings — either via Residual Quantization VAE (RQ-VAE) or hierarchical k-means — and stored in a `.semanticids` file that maps item IDs to code tuples.

The tokenizer is built from these codes. SEM tokens cover all integers $[0, K)$, and each item is tokenized as the corresponding N-token sequence. Crucially, two items that share a code prefix are represented as overlapping token sequences, so the model's residual stream naturally carries information about their shared structure.

### Two-Stage Training

**Pretraining.** The model is pretrained on entity-level random walks over the knowledge graph subgraph (user-item interaction edges excluded). Starting from each entity, random paths of fixed hop length are sampled and formatted as token sequences. Item entities in these paths are expanded to their SEM tuple representation. The objective is standard causal language modeling: next-token prediction over the full sequence. This stage teaches the model the relational structure of the knowledge graph in semantic ID space.

**Finetuning.** The pretrained model is then finetuned on user-specific reasoning paths. Paths are sampled via constrained random walk from each user's positive training items, connecting through knowledge graph entities to reach other items. The path format is `[BOS] U_u R_{ui} SEM_{c_1}...SEM_{c_N} R_{e1} E_{e1} ... R_{eK} SEM_{c_1'}...SEM_{c_N'} [EOS]`, interleaving user, entity, relation, and semantic-item tokens. The finetuning objective is again next-token prediction, but now conditioned on user identity and grounded in actual interaction history.

### Inference

At inference time, the model receives a user token as a prompt and generates beam-search completions. Each completion corresponds to a candidate reasoning path; the terminal item's SEM tuple is decoded via the reverse semantic vocabulary to retrieve the recommended item. The final ranked list is assembled from the unique items appearing across all generated paths, ordered by beam score.

---

## Relation to KGGLM and PEARLM

SPRIG inherits its architecture from PEARLM (which extends KGGLM), using a GPT-2-style causal language model with the same two-stage training pipeline. The sole but consequential difference is the item representation: where KGGLM uses a single item token per item, SPRIG uses N SEM tokens. This change propagates through the tokenizer, the path formatter, the sequence length calculation, and the inference postprocessor, but leaves the training objective and path sampling logic intact. The result is a system where item similarity is encoded in the representation rather than inferred implicitly from training data.

---

## Repository Structure

This is a fork of [hopwise](https://github.com/tail-unica/hopwise). The SPRIG-specific additions relative to the upstream repository are:

- `hopwise/model/path_language_modeling_recommender/sprig.py` — SPRIG model class
- `hopwise/model/sprig_postprocessor.py` — beam output postprocessor for SEM-tuple decoding
- `hopwise/data/dataset/sprig_dataset.py` — dataset class with semantic ID loading, SEM tokenizer, and SEM-aware path formatting
- `hopwise/data/semantic_vocab.py` — semantic vocabulary with forward and reverse lookup, collision handling
- `hopwise/properties/model/SPRIG.yaml` — default configuration

Everything else is inherited from hopwise without modification, including the knowledge graph datasets, path sampling strategies, evaluation pipeline, and all non-SPRIG models.

---

## Installation

Requires Python 3.10 or 3.11 and [uv](https://github.com/astral-sh/uv).

```sh
git clone <this-repo>
cd semantic-id-knowledge-graph-recommender
uv sync
```

---

## Reproducing Results

### 1. Generating Semantic IDs

Semantic IDs must be generated before training. The `.semanticids` file maps each dataset item ID to a tuple of integer codes. Two generation methods are supported: RQ-VAE (preferred, produces contiguous coverage) and k-means over KG entity embeddings.

The file format is a CSV with columns `item_id` and `semantic_ids`:

```
item_id,semantic_ids
1,"[42, 17, 93]"
2,"[42, 17, 8]"
3,"[11, 200, 93]"
```

Item IDs must match the hopwise-internal item IDs for the specific dataset and filter configuration being used. If you change `user_inter_num_interval` or `item_inter_num_interval`, the semantic IDs file must be regenerated.

Place the file at `dataset/<dataset_name>/<dataset_name>.semanticids`.

### 2. Pretraining

```sh
hopwise train --model SPRIG --dataset ml1m \
    --config_files hopwise/properties/model/SPRIG.yaml \
    train_stage=pretrain
```

### 3. Finetuning

```sh
hopwise train --model SPRIG --dataset ml1m \
    --config_files hopwise/properties/model/SPRIG.yaml \
    train_stage=finetune \
    pre_model_path=saved/<pretrained_checkpoint>/
```

---

## Datasets

The MovieLens-1M and Last.FM-1M datasets used in these experiments are available from the [hopwise Google Drive](https://drive.google.com/drive/folders/1Zv57Xfo3mC2DemQbHf5XkYKBluRALQNm?usp=drive_link), as distributed by the upstream hopwise project.

---

## Acknowledgements

This work builds directly on [hopwise](https://github.com/tail-unica/hopwise) by Boratto et al. (CIKM 2025). If you use this codebase, please also cite the upstream library:

```bibtex
@inproceedings{boratto2025hopwise,
  author    = {Boratto, Ludovico and Fenu, Gianni and Marras, Mirko and
               Medda, Giacomo and Soccol, Alessandro},
  title     = {hopwise: A Python Library for Explainable Recommendation
               based on Path Reasoning over Knowledge Graphs},
  booktitle = {Proceedings of the 34th ACM International Conference on
               Information and Knowledge Management},
  series    = {CIKM '25},
  pages     = {6328--6333},
  year      = {2025},
  doi       = {10.1145/3746252.3761641}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
