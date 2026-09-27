from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


NODE_TYPE_DIM = 3


@dataclass
class GraphBatch:
    graph: object
    split: str
    graph_index: int
    graph_id: str
    node_ids: torch.Tensor
    node_type: torch.Tensor
    node_attributes: torch.Tensor
    edge_type: torch.Tensor
    edge_ids: torch.Tensor
    src_ids: torch.Tensor
    dst_ids: torch.Tensor
    timestamps: Optional[torch.Tensor]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    malicious_labels: Optional[torch.Tensor]
    target_edge_mask: torch.Tensor


class ProvFusionDataset:
    """Thin adapter around ProvFusion's merged .pt graph tuple."""

    def __init__(
        self,
        graphs: Dict[str, List[object]],
        reverse_graphs: Optional[Dict[str, List[object]]],
        num_features: int,
        edge_features: int,
        num_classes: int,
        ground_truth_node_ids: Optional[Set[int]],
        source: str,
        max_graphs_per_split: Optional[int] = None,
    ) -> None:
        self.graphs = _limit_graphs(graphs, max_graphs_per_split)
        self.reverse_graphs = _limit_graphs(reverse_graphs, max_graphs_per_split) if reverse_graphs else None
        self.num_features = int(num_features)
        self.edge_features = int(edge_features)
        self.num_classes = int(num_classes)
        self.ground_truth_node_ids = ground_truth_node_ids
        self.source = source

        first = self.first_graph()
        feat = first.ndata["feat"]
        self.node_type_dim = NODE_TYPE_DIM
        self.attr_dim = max(int(feat.shape[1]) - NODE_TYPE_DIM, 0)
        self.edge_type_dim = _edge_type_dim(first)
        self.edge_is_multilabel = _edge_is_multilabel(first.edata["label"])
        self.detection_granularity = "node"

    def first_graph(self) -> object:
        for split in ("train", "val", "test"):
            if self.graphs.get(split):
                return self.graphs[split][0]
        raise ValueError("ProvFusion dataset contains no graphs.")

    def iter_split(self, split: str) -> Iterable[GraphBatch]:
        for graph_index, graph in enumerate(self.graphs.get(split, [])):
            yield make_graph_batch(graph, split, graph_index, self.ground_truth_node_ids)

    def all_splits(self) -> Sequence[str]:
        return tuple(split for split in ("train", "val", "test") if self.graphs.get(split))

    def metadata(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "num_features": self.num_features,
            "num_node_types": self.node_type_dim,
            "node_attribute_dims": [NODE_TYPE_DIM, self.num_features],
            "attribute_dim": self.attr_dim,
            "edge_type_dim": self.edge_type_dim,
            "edge_is_multilabel": self.edge_is_multilabel,
            "detection_granularity": self.detection_granularity,
            "splits": {split: len(self.graphs.get(split, [])) for split in ("train", "val", "test")},
            "ground_truth_available": self.ground_truth_node_ids is not None,
        }


def load_provfusion_dataset(cfg: Dict[str, object]) -> ProvFusionDataset:
    data_cfg = cfg["data"]
    data_path_value = str(data_cfg.get("data_path") or "").strip()
    data_path = Path(data_path_value) if data_path_value else None
    raw_data_dir = Path(data_cfg.get("raw_data_dir") or "../raw_data")
    dataset_name = str(data_cfg.get("dataset_name") or "")

    if data_path is not None and data_path.exists():
        graphs, edge_counts, num_features, edge_features, num_classes, reverse_graphs = _torch_load(data_path)
        source = str(data_path)
    elif dataset_name and (raw_data_dir / dataset_name).is_dir():
        from preprocess import preprocess_dataset

        if data_path is None:
            data_path = Path(f"{dataset_name.lower()}_three_expert_merged.pt")
        preprocess_dataset(dataset_name, str(raw_data_dir), str(data_path))
        graphs, edge_counts, num_features, edge_features, num_classes, reverse_graphs = _torch_load(data_path)
        source = str(data_path)
    elif bool(data_cfg.get("allow_synthetic", False)):
        graphs, edge_counts, num_features, edge_features, num_classes, reverse_graphs = build_synthetic_fixture()
        source = "synthetic-fixture"
    else:
        raise FileNotFoundError(
            "No ProvFusion merged data found. Set data.data_path, provide raw_data_dir/dataset_name, "
            "or enable data.allow_synthetic for smoke tests only."
        )

    ground_truth_value = str(data_cfg.get("ground_truth_path") or "").strip()
    ground_truth_path = Path(ground_truth_value) if ground_truth_value else None
    ground_truth_node_ids = _load_ground_truth(ground_truth_path)
    if source == "synthetic-fixture" and ground_truth_node_ids is None:
        ground_truth_node_ids = {1004}

    return ProvFusionDataset(
        graphs=graphs,
        reverse_graphs=reverse_graphs,
        num_features=num_features,
        edge_features=edge_features,
        num_classes=num_classes,
        ground_truth_node_ids=ground_truth_node_ids,
        source=source,
        max_graphs_per_split=data_cfg.get("max_graphs_per_split"),
    )


def make_graph_batch(
    graph: object,
    split: str,
    graph_index: int,
    ground_truth_node_ids: Optional[Set[int]],
) -> GraphBatch:
    feat = graph.ndata["feat"].float()
    node_type = feat[:, :NODE_TYPE_DIM].argmax(dim=1).long()
    node_attributes = feat[:, NODE_TYPE_DIM:].float()
    node_ids = graph.ndata.get("uuid", torch.arange(graph.num_nodes())).long().cpu()

    src, dst = graph.edges()
    edge_label = graph.edata["label"]
    edge_ids = graph.edata.get("uuid", torch.arange(graph.num_edges())).long().cpu()
    timestamps = graph.edata.get("timestamp", graph.edata.get("time", None))
    malicious_labels = None
    if ground_truth_node_ids is not None:
        malicious_labels = torch.tensor(
            [int(int(node_id) in ground_truth_node_ids) for node_id in node_ids.tolist()],
            dtype=torch.long,
        )

    return GraphBatch(
        graph=graph,
        split=split,
        graph_index=graph_index,
        graph_id=f"{split}:{graph_index}",
        node_ids=node_ids,
        node_type=node_type,
        node_attributes=node_attributes,
        edge_type=edge_label,
        edge_ids=edge_ids,
        src_ids=node_ids[src.cpu()],
        dst_ids=node_ids[dst.cpu()],
        timestamps=timestamps.cpu() if torch.is_tensor(timestamps) else None,
        train_mask=graph.ndata.get("train_mask", torch.zeros(graph.num_nodes(), dtype=torch.bool)).bool().cpu(),
        val_mask=graph.ndata.get("val_mask", torch.zeros(graph.num_nodes(), dtype=torch.bool)).bool().cpu(),
        test_mask=graph.ndata.get("test_mask", torch.zeros(graph.num_nodes(), dtype=torch.bool)).bool().cpu(),
        malicious_labels=malicious_labels,
        target_edge_mask=non_self_edge_mask(graph),
    )


def node_type_one_hot(graph: object) -> torch.Tensor:
    feat = graph.ndata["feat"].float()
    return feat[:, :NODE_TYPE_DIM]


def node_full_features(graph: object) -> torch.Tensor:
    return graph.ndata["feat"].float()


def edge_targets(graph: object, multilabel: bool) -> torch.Tensor:
    labels = graph.edata["label"]
    if labels.ndim == 1:
        return labels.long()
    if multilabel:
        return labels.float()
    return labels.argmax(dim=1).long()


def edge_type_features(graph: object, edge_type_dim: int) -> torch.Tensor:
    labels = graph.edata["label"]
    if labels.ndim == 1:
        return F.one_hot(labels.long(), num_classes=int(edge_type_dim)).float()
    return labels.float()


def non_self_edge_mask(graph: object) -> torch.Tensor:
    src, dst = graph.edges()
    mask = src != dst
    labels = graph.edata.get("label")
    if labels is not None and labels.ndim == 2 and labels.shape[1] > 10:
        mask = mask & (labels.argmax(dim=1) != 10)
    return mask.bool()


def build_synthetic_fixture() -> Tuple[Dict[str, List[object]], torch.Tensor, int, int, int, Dict[str, List[object]]]:
    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - exercised in environments without DGL.
        raise ImportError("Synthetic ProvFusion fixture requires DGL.") from exc

    def make_graph(offset: int) -> object:
        src = torch.tensor([0, 1, 2, 3, 4, 1, 5], dtype=torch.long)
        dst = torch.tensor([1, 2, 3, 4, 5, 5, 0], dtype=torch.long)
        graph = dgl.graph((src, dst), num_nodes=6)
        node_types = F.one_hot(torch.tensor([0, 1, 2, 1, 0, 2]), num_classes=3).float()
        attrs = torch.tensor(
            [
                [1.0, 0.0, 0.2, 0.1],
                [0.9, 0.1, 0.2, 0.0],
                [0.0, 1.0, 0.1, 0.1],
                [0.1, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.9, 0.0],
                [0.0, 0.1, 0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        graph.ndata["feat"] = torch.cat([node_types, attrs], dim=1)
        graph.ndata["label"] = node_types.argmax(dim=1)
        graph.ndata["uuid"] = torch.arange(1000 + offset, 1006 + offset, dtype=torch.long)
        edge_classes = torch.tensor([0, 1, 2, 1, 0, 2, 3], dtype=torch.long)
        graph.edata["label"] = F.one_hot(edge_classes, num_classes=4).float()
        graph.edata["timestamp"] = torch.arange(graph.num_edges(), dtype=torch.long) + offset
        return graph

    graphs = {
        "train": [make_graph(0)],
        "val": [make_graph(10)],
        "test": [make_graph(0)],
    }
    reverse_graphs = {split: [_reverse_graph(g) for g in split_graphs] for split, split_graphs in graphs.items()}
    return graphs, torch.zeros(4), 7, 4, 3, reverse_graphs


def _reverse_graph(graph: object) -> object:
    import dgl

    src, dst = graph.edges()
    new_graph = dgl.graph((dst, src), num_nodes=graph.num_nodes())
    for key, value in graph.ndata.items():
        new_graph.ndata[key] = value
    for key, value in graph.edata.items():
        new_graph.edata[key] = value
    return new_graph


def _limit_graphs(graphs: Optional[Dict[str, List[object]]], limit: Optional[int]) -> Optional[Dict[str, List[object]]]:
    if graphs is None or limit is None:
        return graphs
    return {split: split_graphs[: int(limit)] for split, split_graphs in graphs.items()}


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_ground_truth(path: Optional[Path]) -> Optional[Set[int]]:
    if path is None or not path.exists():
        return None
    obj = _torch_load(path)
    if isinstance(obj, dict):
        if "nids" in obj:
            values = obj["nids"]
        else:
            values = obj.keys()
    else:
        values = obj
    return {int(x) for x in list(values)}


def _edge_type_dim(graph: object) -> int:
    labels = graph.edata["label"]
    if labels.ndim == 1:
        return int(labels.max().item()) + 1 if labels.numel() else 0
    return int(labels.shape[1])


def _edge_is_multilabel(labels: torch.Tensor) -> bool:
    if labels.ndim != 2:
        return False
    positive_counts = (labels > 0).sum(dim=1)
    return bool((positive_counts > 1).any().item())
