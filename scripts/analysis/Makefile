# MetaOthello Analysis Makefile
# Run from the repo root: make -C scripts/analysis <target>
# Or from this directory: make <target>

.PHONY: help all compute figures \
	compute-model-accuracy compute-board-probe-accuracy \
	compute-activation-cosine-sim compute-iago-alignment compute-intervention-eval \
	figures-model-accuracy figures-board-probe-accuracy figures-probe-weight-similarity \
	figures-activation-similarity figures-iago figures-intervention-evaluation \
	compute-divergence-dynamics figures-divergence-dynamics

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-35s\033[0m %s\n", $$1, $$2}'

all: compute figures ## Run all compute and figure scripts in sequence

# ---------------------------------------------------------------------------
# Compute: populate data/analysis_cache/
# ---------------------------------------------------------------------------

compute: ## Run all compute scripts (populate caches)
compute: compute-model-accuracy compute-board-probe-accuracy \
	compute-activation-cosine-sim compute-iago-alignment compute-intervention-eval \
	compute-divergence-dynamics

compute-model-accuracy: ## Compute model accuracy (top1, correct_prob, alpha)
	python model_accuracy/compute_model_accuracy.py --metric top1
	python model_accuracy/compute_model_accuracy.py --metric correct_prob
	python model_accuracy/compute_model_accuracy.py --metric alpha

compute-board-probe-accuracy: ## Compute board probe accuracy per layer
	python board_probe_accuracy/compute_board_probe_accuracy.py

compute-activation-cosine-sim: ## Compute residual-stream activation cosine similarities
	python activation_similarity/compute_activation_cosine_sim.py

compute-iago-alignment: ## Compute Classic-to-Iago Procrustes alignment
	python iago/compute_iago_alignment.py

compute-intervention-eval: ## Compute linear probe intervention evaluation
	python intervention_evaluation/compute_intervention_eval.py

# ---------------------------------------------------------------------------
# Figures: write to figures/<group>/
# ---------------------------------------------------------------------------

figures: ## Run all plot scripts (produce all figures)
figures: figures-model-accuracy figures-board-probe-accuracy \
	figures-probe-weight-similarity figures-activation-similarity \
	figures-iago figures-intervention-evaluation \
	figures-divergence-dynamics

figures-model-accuracy: ## Plot model accuracy figures (over-moves + aggregated, all metrics)
	python model_accuracy/plot_model_accuracy_over_moves.py --metric top1
	python model_accuracy/plot_model_accuracy_over_moves.py --metric correct_prob
	python model_accuracy/plot_model_accuracy_over_moves.py --metric alpha
	python model_accuracy/plot_model_accuracy_aggregated.py --metric top1
	python model_accuracy/plot_model_accuracy_aggregated.py --metric correct_prob
	python model_accuracy/plot_model_accuracy_aggregated.py --metric alpha

figures-board-probe-accuracy: ## Plot board probe accuracy figures (over-moves + aggregated)
	python board_probe_accuracy/plot_board_probe_accuracy_over_moves.py
	python board_probe_accuracy/plot_board_probe_accuracy_aggregated.py

figures-probe-weight-similarity: ## Plot probe weight similarity figures (4 scripts)
	python probe_weight_similarity/plot_tile_state_similarity.py
	python probe_weight_similarity/plot_tile_state_pca.py
	python probe_weight_similarity/plot_cosine_sim_by_layer.py
	python probe_weight_similarity/plot_cosine_sim_aggregated.py

figures-activation-similarity: ## Plot activation cosine similarity over moves
	python activation_similarity/plot_activation_cosine_sim_over_moves.py

figures-iago: ## Plot Iago Procrustes alignment figure
	python iago/plot_iago_alignment.py

figures-intervention-evaluation: ## Plot intervention evaluation figures
	python intervention_evaluation/plot_intervention_cosine.py
	python intervention_evaluation/plot_intervention_global.py

compute-divergence-dynamics: ## Compute divergence dynamics for NoMidFlip (Fig 5)
	python divergence_dynamics/compute_probe_accuracy_differing.py
	python divergence_dynamics/compute_game_probe_fidelity.py
	python divergence_dynamics/compute_steering_nomidflip.py

figures-divergence-dynamics: ## Plot Figure 5 (NoMidFlip divergence dynamics)
	python divergence_dynamics/plot_figure5.py
