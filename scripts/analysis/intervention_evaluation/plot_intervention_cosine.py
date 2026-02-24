"""Plot scatter of cosine similarity vs relative intervention error difference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from metaothello.analysis_utils import CACHE_DIR, load_json_cache
from metaothello.constants import BOARD_DIM
from metaothello.plotting import ICML_HALF_WIDTH, save_figure, setup_icml_style

logger = logging.getLogger(__name__)
FIGURES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "figures" / "intervention_evaluation"
)
N_LAYERS = 8


def load_probe_weights(run_name: str, game_alias: str) -> torch.Tensor:
    """Load probe weights and return as (N_LAYERS, 8, 8, 3, d_model) tensor."""
    probe_dir = CACHE_DIR.parent / run_name / "board_probes"
    probes = []
    for layer in range(1, N_LAYERS + 1):
        probe_path = probe_dir / f"{game_alias}_board_L{layer}.ckpt"
        state_dict = torch.load(probe_path, map_location="cpu")
        weight = state_dict["proj.weight"].detach().reshape(BOARD_DIM, BOARD_DIM, 3, -1)
        probes.append(weight)
    return torch.stack(probes)


def compute_cosine_similarity(probes_1: torch.Tensor, probes_2: torch.Tensor) -> dict[str, float]:
    """Compute cosine similarity between probe weights for each (tile, state).

    Averages across all layers.
    Returns dict mapping 'tile_state' -> cos_sim.
    """
    state_names = {0: "yours", 1: "empty", 2: "mine"}
    similarities = {}

    for r in range(BOARD_DIM):
        for c in range(BOARD_DIM):
            tile_id = chr(ord("a") + c) + str(r + 1)
            for state_idx, state_name in state_names.items():
                key = f"{tile_id}_{state_name}"
                cos_sims = []

                for layer in range(N_LAYERS):
                    vec_1 = probes_1[layer, r, c, state_idx]
                    vec_2 = probes_2[layer, r, c, state_idx]
                    norm_1, norm_2 = torch.norm(vec_1), torch.norm(vec_2)

                    if norm_1 > 1e-8 and norm_2 > 1e-8:
                        sim = float(torch.dot(vec_1, vec_2) / (norm_1 * norm_2))
                        cos_sims.append(sim)

                similarities[key] = float(np.mean(cos_sims)) if cos_sims else float("nan")

    return similarities


def compute_intervention_diff(correct_results: dict, cross_results: dict) -> dict[str, float]:
    """Compute relative error increase: (cross_error - correct_error) / correct_error."""
    diffs = {}
    for key in correct_results:
        if key not in cross_results:
            continue

        c_vals = correct_results[key]
        x_vals = cross_results[key]

        if len(c_vals) < 5 or len(x_vals) < 5:
            continue

        _, _, correct_lin, _, correct_n = c_vals
        _, _, cross_lin, _, cross_n = x_vals

        if correct_n == 0 or cross_n == 0 or np.isnan(correct_lin) or np.isnan(cross_lin):
            continue

        if abs(correct_lin) < 1e-6:
            continue

        diffs[key] = (cross_lin - correct_lin) / correct_lin

    return diffs


def plot_cosine_vs_diff(cache: dict) -> None:
    """Create and save the scatter plot."""
    setup_icml_style()

    models = ["classic_nomidflip", "classic_delflank"]
    models = [m for m in models if m in cache]
    if not models:
        logger.error("No valid models found in cache.")
        return

    title_dict = {
        "classic_nomidflip": "Classic vs NoMidFlip",
        "classic_delflank": "Classic vs DelFlank",
    }

    state_colors = {"mine": "#4682b4", "yours": "#ff7f50", "empty": "#2e8b57"}
    state_markers = {"mine": "o", "yours": "s", "empty": "^"}

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(ICML_HALF_WIDTH, ICML_HALF_WIDTH * 0.7),
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=False):
        model_results = cache[model]
        g1, g2 = model.split("_")

        try:
            probes_1 = load_probe_weights(model, g1)
            probes_2 = load_probe_weights(model, g2)
        except Exception as e:
            logger.warning("Could not load probes for %s: %s", model, e)
            ax.text(0.5, 0.5, "Probe loading failed", transform=ax.transAxes, ha="center")
            continue

        cos_sims = compute_cosine_similarity(probes_1, probes_2)

        all_cos, all_diff, all_states = [], [], []

        for game in [g1, g2]:
            correct_key, cross_key = f"{game}_correct", f"{game}_cross"
            if correct_key not in model_results or cross_key not in model_results:
                continue

            diffs = compute_intervention_diff(model_results[correct_key], model_results[cross_key])

            for key, diff_val in diffs.items():
                if key in cos_sims and not np.isnan(cos_sims[key]) and not np.isnan(diff_val):
                    state = key.split("_")[1]
                    all_cos.append(cos_sims[key])
                    all_diff.append(diff_val)
                    all_states.append(state)

        # Plot by state
        for state in ["mine", "yours", "empty"]:
            idx = [i for i, s in enumerate(all_states) if s == state]
            if idx:
                ax.scatter(
                    [all_cos[i] for i in idx],
                    [all_diff[i] for i in idx],
                    c=state_colors[state],
                    marker=state_markers[state],
                    label=state.capitalize(),
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )

        if all_cos and all_diff:
            z = np.polyfit(all_cos, all_diff, 1)
            x_line = np.linspace(min(all_cos), max(all_cos), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.6, linewidth=1.0)

            corr = float(np.corrcoef(all_cos, all_diff)[0, 1])
            ax.annotate(
                f"$r = {corr:.3f}$",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "gray",
                    "alpha": 0.9,
                },
            )

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Cosine Similarity (Probe$_1$ vs Probe$_2$)")
        if model == models[0]:
            ax.set_ylabel(r"Rel. $\Delta$ Error (Cross - Correct) / Correct")
        ax.set_title(title_dict.get(model, model))

        if model == models[0]:
            ax.legend(loc="upper right", frameon=False)

    save_figure(fig, FIGURES_DIR / "cosine_vs_intervention_diff")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cosine similarity vs intervention diff.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cache_file = CACHE_DIR / "intervention_eval.json"
    cache = load_json_cache(cache_file)
    if not cache:
        logger.error("Cache file not found or empty: %s", cache_file)
    else:
        plot_cosine_vs_diff(cache)
