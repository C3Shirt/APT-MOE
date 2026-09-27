import torch
import pytest


def test_four_experts_forward_backward_on_synthetic_fixture():
    pytest.importorskip("dgl")

    from apt_moe.data.provfusion_adapter import (
        build_synthetic_fixture,
        edge_targets,
        edge_type_features,
        node_full_features,
        node_type_one_hot,
        non_self_edge_mask,
    )
    from apt_moe.models import CausalityExpert, GraphExpert, NormalExpert, SemanticExpert

    graphs, _, num_features, edge_features, _, _ = build_synthetic_fixture()
    graph = graphs["train"][0]
    node_type = node_type_one_hot(graph)
    full = node_full_features(graph)
    semantic = full[:, 3:]
    edge_feat = edge_type_features(graph, edge_features)
    edge_idx = torch.nonzero(non_self_edge_mask(graph), as_tuple=False).view(-1)
    targets = edge_targets(graph, True)
    mask_nodes = torch.tensor([0, 2, 4])
    expert_dim = 6

    semantic_model = SemanticExpert(semantic_dim=semantic.shape[1], hidden_dim=8, expert_dim=expert_dim, dropout=0.0)
    semantic_out = semantic_model(semantic, target_mask=mask_nodes)
    assert semantic_out.residual.shape == (graph.num_nodes(), expert_dim)
    assert semantic_out.score.shape == (graph.num_nodes(),)
    assert torch.isfinite(semantic_out.loss)
    semantic_out.loss.backward()

    normal_model = NormalExpert(input_dim=num_features, hidden_dim=8, expert_dim=expert_dim, dropout=0.0)
    normal_model.set_prototype(full.mean(dim=0))
    normal_out = normal_model(full)
    assert normal_out.residual.shape == (graph.num_nodes(), expert_dim)
    assert normal_out.score.shape == (graph.num_nodes(),)
    assert torch.isfinite(normal_out.loss)
    normal_out.loss.backward()

    graph_model = GraphExpert(
        node_type_dim=3,
        edge_type_dim=edge_features,
        hidden_dim=8,
        expert_dim=expert_dim,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    graph_out = graph_model(graph, node_type, edge_feat, mask_nodes=mask_nodes)
    assert graph_out.residual.shape == (graph.num_nodes(), expert_dim)
    assert graph_out.score.shape == (graph.num_nodes(),)
    assert torch.isfinite(graph_out.loss)
    graph_out.loss.backward()

    causal_model = CausalityExpert(
        node_feat_dim=num_features,
        edge_type_dim=edge_features,
        hidden_dim=8,
        expert_dim=expert_dim,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        multilabel=True,
    )
    causal_out = causal_model(graph, full, targets, edge_feat, edge_indices=edge_idx)
    assert causal_out.residual.shape == (graph.num_nodes(), expert_dim)
    assert causal_out.score.shape == (graph.num_nodes(),)
    assert torch.isfinite(causal_out.loss)
    causal_out.loss.backward()


def test_graph_and_causality_encoders_do_not_share_parameters():
    from apt_moe.models import CausalityExpert, GraphExpert

    graph_model = GraphExpert(
        node_type_dim=3,
        edge_type_dim=4,
        hidden_dim=8,
        expert_dim=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    causal_model = CausalityExpert(
        node_feat_dim=7,
        edge_type_dim=4,
        hidden_dim=8,
        expert_dim=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        multilabel=True,
    )
    graph_param_ids = {id(param) for param in graph_model.encoder.parameters()}
    causal_param_ids = {id(param) for param in causal_model.encoder.parameters()}
    assert graph_param_ids
    assert causal_param_ids
    assert graph_param_ids.isdisjoint(causal_param_ids)
