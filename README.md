

<h1 align="center">SPRIG (anonymous submission)</h1>
<p align="center">
    <b>Implementation of the SPRIG paper, built on top of hopwise.</b>
</p>

## Overview

This repository contains the implementation of **SPRIG**, a generative recommendation model that combines **Semantic IDs (SIDs)** with **knowledge graph path reasoning**. It is built on top of the hopwise codebase (see the original repository at https://github.com/tail-unica/hopwise).

**Abstract highlights (SPRIG):**
- Recent KG-augmented sequential recommenders still represent items as opaque identifiers tied to large embedding tables, limiting parameter sharing and generalization.
- SPRIG represents items as short sequences of discrete codes obtained via hierarchical quantization of item content features, enabling cross-item generalization and compressing the item vocabulary.
- SPRIG combines the strengths of SIDs with the expressiveness of KG path reasoning and achieves competitive performance with substantially fewer parameters.
- To the best of our knowledge, SPRIG is the first model to integrate Semantic IDs with KG path reasoning for generative recommendation.

**Related work:** Prior KG path reasoning approaches such as KGGLM demonstrate the effectiveness of reasoning over structured paths for recommendation and also prior works like TIGER showcases that generative recommendations can be improved using hierarchical semantically charged identifiers. SPRIG builds on these ideas while replacing item ID embeddings with discrete, shareable semantic codes to improve generalization and reduce parameter count.

---

![hopwise pipeline](https://github.com/tail-unica/hopwise/blob/main/assets/hopwise.png)

## Installation

Prerequisites:
- Python **3.9**, **3.10**, or **3.11**
- [`uv`](https://github.com/astral-sh/uv) package manager

Create the environment and install dependencies from source:
```sh
uv venv --python PYTHON_VERSION --prompt sprig
uv sync
```

## How to Execute (SPRIG)

### 1) Train
```sh
uv run -q hopwise --debug train \
    --model SPRIG \
    --dataset DATASET \
    --config_files ./hopwise/properties/SPRIG-pretrain.yaml
```

```sh
uv run -q hopwise --debug train \
    --model SPRIG \
    --dataset DATASET \
    --config_files ./hopwise/properties/SPRIG-fientune.yaml
```

Override config parameters directly from the CLI using =:
```sh
uv run -q hopwise train --model SPRIG --dataset DATASET --epochs=20
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

