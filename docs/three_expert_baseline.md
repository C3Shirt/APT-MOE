# Three-Expert Self-Supervised APT Baseline

This document records the ProvFusion audit and the implementation boundary for
`apt_moe/`. The current goal is not a full MoE router; it is a runnable baseline
that asks whether node-type, edge-type, and attribute self-supervision produce
complementary anomaly signals on the same test split.

## Scope

- Main trunk: ProvFusion only.
- External code: KAIROS and CAMERA were only used as directory context; no PyG or
  KAIROS code is imported into the new path.
- Original entry points: `main_transductive.py`, `try_different_threshold.py`,
  and GraphMAE source files are not modified.
- Data downloads: disabled. The user-provided raw DARPA JSON root is
  `F:\phd\data\APT\DARPA`. This baseline will not download Hugging Face assets.

## Repository State

`F:\phd\project\apt-project\relative_code\ProvFusion_Private-main` is not a Git
repository in this workspace, and neither parent `apt-project` nor `relative_code`
contains a `.git` directory. Therefore `git status` cannot identify user edits in
this copy. All new work is confined to `apt_moe/`, `configs/`, `docs/`, and
`tests/`.

## ProvFusion Data Audit

Audit target files:

- `graphmae/datasets/data_util.py`
- `preprocess.py`
- `main_transductive.py`
- `graphmae/models/edcoder.py`
- `graphmae/models/gatedge.py`
- `link_prediction_evaluation.py`
- `try_different_threshold.py`

Findings from code inspection:

1. Node types: exactly 3 classes. `convert_to_dgl_data()` sets
   `node_type_dim = 3`, stores one-hot node type in `feat[:, :3]`, and stores
   `graph.ndata["label"] = argmax(node_type)`.
2. `feat[:, :3]` is node type. `main_transductive.py` also explicitly truncates
   `graph.ndata["feat"] = graph.ndata["feat"][:, :3]` for the pure-structure
   view.
3. Node attributes are `feat[:, 3:]`. In the default conversion this is the
   128-dimensional node embedding from the source/destination message fields, so
   full node feature dimensionality is expected to be 131.
4. Edge types are multi-hot after graph conversion. Raw event type is one-hot over
   11 classes in `merge_multi_edges_with_reverse()`, then repeated edges between
   the same `(src, dst)` are collapsed with multi-hot max aggregation. Self-loops
   are assigned class 10. Because collapsed edges may contain more than one
   positive type, the adapter detects whether each concrete dataset is single- or
   multi-label.
5. Ground truth detection granularity is node-level. README says detection is
   node-level; evaluation loads `ground_truth_nids.pt` and compares
   `graph.ndata["uuid"]` against those node ids. Original edge-reconstruction
   losses are aggregated back to nodes by max over incident endpoints.
6. The original `GATEdge` encoder reads edge labels in attention when
   `edge_in_dim > 0`. That is appropriate for the released ProvFusion pipeline but
   unsafe for Edge-Type Prediction, because the target edge type can become an
   encoder input. The new `apt_moe` edge expert therefore uses `SafeGraphEncoder`,
   which accepts only graph structure and node features.
7. Splits are directory-derived: `Custimized()` loads `train/`, `val/`, and
   `test/` directories and sets node masks accordingly. Code inspection cannot
   prove chronological or entity leakage safety for a future dataset without the
   actual merged data or raw embedding directories. This must be rechecked when
   real data is attached.

## New Implementation

The new package adds:

- `apt_moe/data/provfusion_adapter.py`: wraps the existing ProvFusion merged
  `.pt` tuple and exposes graph, node type, node attributes, edge type, ids,
  split masks, timestamps when available, and node-level malicious labels.
- `apt_moe/models/node_type_expert.py`: masked node-type prediction. The target
  node type is removed from input before prediction.
- `apt_moe/models/edge_type_expert.py`: edge-type prediction from source and
  destination embeddings. It never receives edge type as input.
- `apt_moe/models/attribute_expert.py`: masked attribute prediction. Node type is
  retained, target attribute dimensions are masked.
- `apt_moe/evaluation/`: empirical CDF calibration, CSV export, and
  complementarity analysis.
- `apt_moe/train_experts.py` and `apt_moe/evaluate_experts.py`: module entry
  points.

Because ProvFusion's confirmed ground truth is node-level, this first baseline
exports one row per node. Edge expert scores are mapped to nodes using the same
max-over-incident-edges idea used by the original evaluation path. The CSV keeps
the requested event-style columns, but for node-level rows `src_id == dst_id ==
sample_id`, and `node_event_raw`/`attr_event_raw` equal the corresponding node
score.

## Calibration and Leakage Boundary

- Training and early stopping use only self-supervised train/validation losses.
- Test labels are never used for training, early stopping, or calibration.
- Empirical CDF calibration uses the configured calibration split, default `val`,
  and filters to benign samples when labels are available.
- If labels are unavailable, calibration still runs on the configured split but
  the output should be treated as unlabeled smoke output, not a paper result.

## Data Boundary

`configs/three_expert_smoke.yaml` uses an in-code synthetic DGL fixture for
smoke tests only.

`configs/three_expert_full.yaml` is prepared for local data only:

- set `data.data_path` to a ProvFusion merged `.pt` file; or
- set `data.raw_data_dir` to a ProvFusion `raw_data` directory containing
  `{DATASET}/train`, `{DATASET}/val`, and `{DATASET}/test` embedding graphs.

The local raw DARPA JSON root `F:\phd\data\APT\DARPA` is not itself a
ProvFusion merged-data directory. Building embeddings from it requires the
separate heavy `data_preparation` pipeline and PostgreSQL; this baseline does
not run that automatically.

## Commands

Smoke code path, once DGL is available:

```bash
python -m apt_moe.train_experts --config configs/three_expert_smoke.yaml
python -m apt_moe.evaluate_experts --config configs/three_expert_smoke.yaml
```

Real data, after setting `data.data_path` or `data.raw_data_dir` in
`configs/three_expert_full.yaml`:

```bash
python -m apt_moe.train_experts --config configs/three_expert_full.yaml
python -m apt_moe.evaluate_experts --config configs/three_expert_full.yaml
```

Optional single-expert training:

```bash
python -m apt_moe.train_experts --config configs/three_expert_full.yaml --experts node
python -m apt_moe.train_experts --config configs/three_expert_full.yaml --experts edge
python -m apt_moe.train_experts --config configs/three_expert_full.yaml --experts attr
```

## Current Environment Note

The default local `python` observed during implementation is Python 3.14 with
CPU PyTorch but without DGL. ProvFusion's README pins Python 3.9 and DGL 1.0.0.
In this environment, DGL-dependent forward/backward tests are skipped; they
should be rerun in the ProvFusion environment before real experiments.
