# Joint Four-Expert APT-MoE Baseline

This implementation follows the end-to-end training scope in GitHub issue #2.
All four experts and the gate are initialized together and updated from the
first epoch by one Adam optimizer. There is no expert pretraining, freezing,
pseudo-anomaly generation, or supervised attack-label loss in the canonical
pipeline.

## Model and Self-Supervised Objectives

- `SemanticExpert` reconstructs the 128D semantic feature and uses per-node
  mean squared reconstruction error.
- `GraphExpert` masks node-type features and reconstructs the 3D node type;
  cross-entropy over masked nodes is its objective and anomaly energy.
- `NormalExpert` decodes the global 131D mean of training nodes. The prototype
  is computed without labels and stored in the checkpoint; the decoder is
  trained with full-feature reconstruction MSE.
- `CausalityExpert` predicts edge/event types with BCE for multilabel targets
  or cross-entropy otherwise. Node energy is the temperature-controlled
  log-sum-exp over incident edge losses; exact incident maximum is also
  exported for diagnosis.

Each expert energy is standardized using detached EMA mean and standard
deviation, then passed through softplus. The gate sees the detached 131D node
feature concatenated with the mean incoming-neighbor feature; it has no GNN.
Its softmax weights fuse the four nonnegative energies. The final score is
`sigmoid(fused_energy)`.

The jointly optimized objective is

```text
L = sum_k lambda_k L_k
    + alpha * H(gate)
    + eta * KL(mean_gate || uniform)
    + beta * BCE(sigmoid(fused_energy), 0)
```

The entropy term is minimized to encourage sharp routing. The optional global
load-balancing coefficient `eta` defaults to zero in the model; the full
CADETS_E3 run sets it to 0.02 to prevent population-level expert collapse.
Configured training-graph indices can be held out and combined with the
official validation split; held-out graphs are excluded from optimizer updates
and from the normal prototype. The final one-class BCE
assumes the unlabeled training nodes are predominantly benign. Its training
term is evaluated on nodes whose graph expert energy was produced by masking
that node in the current step. Every component and the weighted total are
logged each epoch. Loss coefficients and EMA settings live in the `moe` and
`loss_weights` config sections.

## Thresholding and Labels

No attack labels are used by the expert losses, gate loss, or early stopping.
The anomaly threshold is the configured quantile (default 0.999) of validation
fused energies; if validation labels exist, only benign validation nodes are
used to estimate it. Test labels are used only in the final offline report.
Without labels, the report still provides detection counts, expert/gate
diagnostics, and score exports, while supervised metrics remain unavailable.

## Causal Edge Context

Issue #2 treats target-edge leakage prevention as outside its scope. Therefore
the causal encoder currently receives the original edge-type features,
including the relation being predicted. The fold masking applied at evaluation
is for the graph node-reconstruction expert only. Causal target-edge leakage
must be addressed in a separate change before interpreting causal scores as
strict held-out edge predictions.

## Run

```bash
python -m apt_moe.train_experts --config configs/four_expert_smoke.yaml
python -m apt_moe.evaluate_experts --config configs/four_expert_smoke.yaml
```

The smoke configuration enables a tiny synthetic graph for code-path checks.
For ProvFusion data, set `data.data_path` in
`configs/four_expert_full.yaml` to the local merged graph file and run the same
two commands. Training writes one `apt_moe_joint.pt` checkpoint.
