# MetaOthello: A Controlled Study of Multiple World Models in Transformers

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Models%20%26%20Data-orange)](https://huggingface.co/aviralchawla/metaothello)

> **Paper:** [MetaOthello: A Controlled Study of Multiple World Models in Transformers](pending_link) — *Arxiv*

---

## Abstract

Foundation models must handle multiple generative processes, yet mechanistic interpretability largely studies capabilities in isolation; it remains unclear how a single transformer organizes multiple, potentially conflicting "world models". Previous experiments on Othello-playing neural networks test world-model learning but focus on a single game with a single set of rules. We introduce *MetaOthello*, a controlled suite of Othello variants with shared syntax but different rules or tokenizations, and train small GPTs on mixed-variant data to study how multiple world models are organized in a shared representation space. We find that transformers trained on mixed-game data do not partition their capacity into isolated sub-models; instead, they converge on a mostly shared board-state representation that transfers causally across variants. Linear probes trained on one variant can intervene on another's internal state with effectiveness approaching that of matched probes. For isomorphic games with token remapping, representations are equivalent up to a single orthogonal rotation that generalizes across layers. When rules partially overlap, early layers maintain game-agnostic representations while a middle layer identifies game identity, and later layers specialize. MetaOthello offers a path toward understanding not just whether transformers learn world models, but how they organize many at once.

---

## Table of Contents

1. [Installation](#installation)
2. [The `metaothello` Package](#the-metaothello-package)
   - [Game Engine](#game-engine)
   - [Game Variants](#game-variants)
   - [Basic Usage](#basic-usage)
3. [Downloading Pretrained Assets](#downloading-pretrained-assets)
4. [Generating Data from Scratch](#generating-data-from-scratch)
5. [Training a GPT Model](#training-a-gpt-model)
6. [Training Board Probes](#training-board-probes)
7. [Reproducing Analysis and Figures](#reproducing-analysis-and-figures)
8. [Code Quality](#code-quality)
9. [Citation](#citation)

---

## Installation

**Requirements:** Python 3.12+, PyTorch 2.1+

```bash
# 1. Clone the repository
git clone https://github.com/aviralchawla/metaothello.git
cd metaothello

# 2. Install the package and all dependencies
pip install -e .

# 3. (Optional) Install dev tools — linting, type-checking, testing
pip install -e ".[dev]"
pre-commit install
```

Key dependencies: `torch`, `transformer_lens`, `xarray`, `zarr`, `matplotlib`, `seaborn`, `scipy`, `huggingface_hub`.

---

## The `metaothello` Package

### Game Engine

Each game variant is defined by three composable, stateless rule classes:

| Rule Type | Role |
|---|---|
| `InitializeBoard` | Sets the starting board configuration |
| `ValidationRule` | Determines which moves are legal at each step |
| `UpdateRule` | Applies a move to the board (flip, delete, etc.) |

### Game Variants

| Variant | Alias | Update Rule | Key Difference |
|---|---|---|---|
| **Classic** | `classic` | Flip all flanked pieces | Standard Othello |
| **NoMidFlip** | `nomidflip` | Flip endpoints of flanked sequence only | High game-tree overlap with Classic |
| **DelFlank** | `delflank` | Delete flanked pieces; open-spread init; neighbor validation | Very different from Classic |
| **Iago** | `iago` | Flip all flanked pieces | Identical to Classic but with a scrambled token vocabulary |

Iago serves as the **isomorphic control**: board squares are mapped to tokens via a fixed permutation, so the model must learn the same latent structure through a different surface vocabulary.

### Basic Usage

```python
from metaothello.games import ClassicOthello

# Instantiate any game by alias
game = ClassicOthello()

# Play a game
print(game.get_valid_moves())   # e.g. ['c4', 'd3', 'e6', 'f5']

game.play_move("d3")
game.print_board()

# Retrieve history
board_history = game.get_board_history()  # List of (8, 8) snapshots
move_history  = game.get_history()        # List of move name strings
```

```python
# List all registered game aliases
list(GAME_REGISTRY.keys())
# ['classic', 'nomidflip', 'delflank', 'iago']
```

```python
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.analysis_utils import gen_games

tokenizer = Tokenizer()

# Generate 100 complete Classic games (each exactly 60 moves)
# Returns: seqs (100, 60) int32 token IDs
#          valid_masks (100, 60, 66) bool — valid next-move mask at each step
seqs, valid_masks = gen_games("classic", num_games=100, tokenizer=tokenizer)
```

---

## Downloading Pretrained Assets

All pretrained models, board probes, and training data are hosted on HuggingFace:

- **Models & probes:** [`aviralchawla/metaothello`](https://huggingface.co/aviralchawla/metaothello)
- **Datasets:** [`datasets/aviralchawla/metaothello`](https://huggingface.co/datasets/aviralchawla/metaothello)

### Download everything at once

```bash
make download-all        # GPT checkpoints + training data + board probes
```

### Download selectively

```bash
# GPT checkpoints
make download-models                           # All 7 model checkpoints
make download-model RUN_NAME=classic           # Single model

# Training data (Zarr stores)
make download-data                             # All game datasets
make download-data-game GAME=classic           # Single game

# Board probe checkpoints
make download-all-probes                       # All runs (8 probes × games per run)
make download-probe-single RUN_NAME=classic    # Single run's probes
```

Assets are placed under `data/` following this layout:

```
data/
├── classic/
│   └── train_classic_20M.zarr
├── classic_nomidflip/
│   ├── train_config.json
│   ├── ckpts/
│   │   └── epoch_250.ckpt
│   └── board_probes/
│       ├── classic_board_L1.ckpt   # through L8
│       └── nomidflip_board_L1.ckpt # through L8
...
```

### Stream data without downloading

```python
import xarray as xr
ds = xr.open_zarr("hf://datasets/aviralchawla/metaothello/train_classic_20M.zarr")
```

---

## Generating Data from Scratch

To generate game data locally instead of downloading:

```bash
# Generate N million games for a single variant  [defaults: GAME=classic, N_GAMES=1, SPLIT=train]
make generate-data GAME=classic N_GAMES=20 SPLIT=train

# Generate training data for all four games (20M each)
make generate-data-all-train
```

Data is written to `data/{game}/{split}_{game}_{N}M.zarr`.

For probe training, also generate a 1M probe-training set per game:

```bash
make generate-data GAME=classic N_GAMES=1 SPLIT=board_train
```

---

## Training a GPT Model

### 1. Prepare the training config

Each run reads hyperparameters from `data/{run_name}/train_config.json`. Configs are included for all 7 paper runs.

**Single-game:**
```json
{
    "run_name": "classic",
    "data": [
        {"game": "classic", "num_games": 20, "path": "data/classic/train_classic_20M.zarr"}
    ],
    "training": {
        "max_epochs": 250,
        "batch_size": 4096,
        "learning_rate": 0.0005,
        "lr_decay": true,
        "betas": [0.9, 0.95],
        "grad_norm_clip": 1.0,
        "weight_decay": 0.1
    }
}
```

**Mixed-game** — add both datasets under `"data"`:
```json
{
    "run_name": "classic_nomidflip",
    "data": [
        {"game": "classic",   "num_games": 20, "path": "data/classic/train_classic_20M.zarr"},
        {"game": "nomidflip", "num_games": 20, "path": "data/nomidflip/train_nomidflip_20M.zarr"}
    ],
    ...
}
```

### 2. Run training

```bash
# Via Makefile
make train RUN_NAME=classic

# Directly
python scripts/gpt_train.py --run_name classic
python scripts/gpt_train.py --run_name classic_nomidflip --verbose
```

Training **automatically resumes** from the latest checkpoint in `data/{run_name}/ckpts/`. Checkpoints are saved each epoch as `epoch_{N}.ckpt`. Fixed architecture: 8 layers, d_model=512, 8 heads, seed=42.

---

## Training Board Probes

Board probes are linear classifiers trained to predict the tile state (Mine / Opponent / Empty) at each board position from the model's residual-stream activations. One probe is trained per `(model, game, layer)` triple — 8 probes for single-game models, 16 for mixed-game models.

### Step 1: Cache activations

Probes train on cached activations. Run the caching script once per `(model, data)` pair:

```bash
# Via Makefile
make cache-activations RUN_NAME=classic DATA_PATH=data/classic/board_train_classic_1M.zarr

# Directly
python scripts/cache_activations.py \
    --run_name classic \
    --data_path data/classic/board_train_classic_1M.zarr
```

This appends `resid_post` activations for all 8 layers into the Zarr store. For mixed-game models, cache activations separately for each constituent game's data.

### Step 2: Train probes

```bash
# Via Makefile  [MODEL_NAME=classic PROBE_GAME=classic LAYER=1]
make train-probe MODEL_NAME=classic PROBE_GAME=classic LAYER=5

# Directly
python scripts/board_probe_train.py \
    --model_name classic \
    --game classic \
    --layer 5
```

**Train all 8 layers for a single-game model:**
```bash
for layer in $(seq 1 8); do
    python scripts/board_probe_train.py --model_name classic --game classic --layer $layer
done
```

**Train all 16 probes for a mixed-game model:**
```bash
for game in classic nomidflip; do
    for layer in $(seq 1 8); do
        python scripts/board_probe_train.py \
            --model_name classic_nomidflip --game $game --layer $layer
    done
done
```

Probes are saved to `data/{model_name}/board_probes/{game}_board_L{layer}.ckpt` (layers 1-indexed, 10 training epochs, lr=3e-4).

---

## Reproducing Analysis and Figures

All analysis follows a **compute → plot** split: compute scripts run GPU inference and write results to `data/analysis_cache/`; plot scripts are CPU-only and write figures to `figures/`. The `scripts/analysis/Makefile` orchestrates the full pipeline.

### Step 1: Populate caches (GPU required)

```bash
make -C scripts/analysis compute
```

| Sub-target | Cache file |
|---|---|
| `compute-model-accuracy` | `data/analysis_cache/model_accuracy.json` |
| `compute-board-probe-accuracy` | `data/analysis_cache/board_probe_accuracy.pkl` |
| `compute-activation-cosine-sim` | `data/analysis_cache/activation_cosine_sim.json` |
| `compute-iago-alignment` | `data/analysis_cache/iago_alignment.json` |
| `compute-intervention-eval` | `data/analysis_cache/intervention_eval.json` |

### Step 2: Generate all figures (CPU only)

```bash
make -C scripts/analysis figures
```

| Sub-target | Figures produced | Output directory |
|---|---|---|
| `figures-model-accuracy` | Accuracy over moves + aggregated (×3 metrics) | `figures/model_accuracy/` |
| `figures-board-probe-accuracy` | Probe accuracy over moves + aggregated | `figures/board_probe_accuracy/` |
| `figures-probe-weight-similarity` | Tile-state heatmaps, PCA, cosine sim by layer + aggregated | `figures/probe_weight_similarity/` |
| `figures-activation-similarity` | Activation cosine similarity over moves | `figures/activation_similarity/` |
| `figures-iago` | Classic-to-Iago Procrustes alignment | `figures/iago/` |
| `figures-intervention-evaluation` | Global intervention errors + cosine vs. error scatter | `figures/intervention_evaluation/` |

> **Note:** `figures-probe-weight-similarity` loads probe checkpoints directly from `data/` — no compute step is needed for these figures.

### Run everything end to end

```bash
make -C scripts/analysis all     # compute then figures in sequence
make -C scripts/analysis help    # list all targets with descriptions
```

All figures are saved as both PDF and PNG at 300 DPI.

---

## Citation

If you use MetaOthello in your research, please cite:

```bibtex
@article{metaothello2025,
  title   = {MetaOthello: A Controlled Study of Multiple World Models in Transformers},
  author  = {Anonymous Authors},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
