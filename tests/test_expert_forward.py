import torch
import pytest


def test_three_experts_forward_backward_on_synthetic_fixture():
    dgl = pytest.importorskip("dgl")  # noqa: F841

    from apt_moe.data.provfusion_adapter import (
        build_synthetic_fixture,
        edge_targets,
        node_full_features,
        node_type_one_hot,
        non_self_edge_mask,
    )
    from apt_moe.models import AttributeExpert, EdgeTypeExpert, NodeTypeExpert

    graphs, _, num_features, edge_features, _, _ = build_synthetic_fixture()
    graph = graphs["train"][0]
    node_type = node_type_one_hot(graph)
    full = node_full_features(graph)
    attrs = full[:, 3:]
    mask_nodes = torch.tensor([0, 2, 4])

    node_model = NodeTypeExpert(num_node_types=3, hidden_dim=8, num_layers=1, dropout=0.0)
    node_loss = node_model.loss(graph, node_type, node_type.argmax(dim=1), mask_nodes)
    assert torch.isfinite(node_loss)
    node_loss.backward()

    attr_model = AttributeExpert(node_type_dim=3, attr_dim=attrs.shape[1], hidden_dim=8, num_layers=1, dropout=0.0)
    attr_loss = attr_model.loss(graph, node_type, attrs, mask_nodes)
    assert torch.isfinite(attr_loss)
    attr_loss.backward()

    src, dst = graph.edges()
    all_edges = torch.stack([src, dst], dim=1)
    edge_idx = torch.nonzero(non_self_edge_mask(graph), as_tuple=False).view(-1)
    edge_model = EdgeTypeExpert(
        node_feat_dim=num_features,
        edge_type_dim=edge_features,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        multilabel=False,
    )
    edge_loss = edge_model.loss(graph, full, all_edges[edge_idx], edge_targets(graph, False)[edge_idx])
    assert torch.isfinite(edge_loss)
    edge_loss.backward()

    node_scores = node_model.node_scores(graph, node_type, node_type.argmax(dim=1), mask_folds=3)
    attr_scores = attr_model.node_scores(graph, node_type, attrs, mask_folds=3)
    edge_scores = edge_model.edge_scores(graph, full, all_edges[edge_idx], edge_targets(graph, False)[edge_idx])
    assert node_scores.shape[0] == graph.num_nodes()
    assert attr_scores.shape[0] == graph.num_nodes()
    assert edge_scores.shape[0] == edge_idx.numel()
