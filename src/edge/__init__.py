"""Week 7: the frozen mask+tracker(hist=15) pipeline under the constraints of a wrist device -
a causal filter, float32 and fixed-point arithmetic, duty-cycled bursts - and the inference
budget that follows (docs/edge-budget.md). Nothing here is fitted: every hyperparameter is the
one each LOSO fold selected on its training subjects (results/edge_fold_configs.csv)."""
