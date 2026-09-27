import torch

from apt_moe.models import CausalityExpert, GraphExpert, NormalExpert, ProvFusionGATEdgeEncoder, SemanticExpert


def test_semantic_and_normal_experts_are_graph_independent():
    semantic = SemanticExpert(semantic_dim=4, hidden_dim=8, expert_dim=6, dropout=0.0)
    normal = NormalExpert(input_dim=7, hidden_dim=8, expert_dim=6, dropout=0.0)
    assert semantic.uses_gnn is False
    assert normal.uses_gnn is False
    assert not any(isinstance(module, ProvFusionGATEdgeEncoder) for module in semantic.modules())
    assert not any(isinstance(module, ProvFusionGATEdgeEncoder) for module in normal.modules())


def test_graph_and_causality_use_gatedge_and_edge_type_features():
    graph = GraphExpert(
        node_type_dim=3,
        edge_type_dim=4,
        hidden_dim=8,
        expert_dim=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    causal = CausalityExpert(
        node_feat_dim=7,
        edge_type_dim=4,
        hidden_dim=8,
        expert_dim=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        multilabel=True,
    )
    assert isinstance(graph.encoder, ProvFusionGATEdgeEncoder)
    assert isinstance(causal.encoder, ProvFusionGATEdgeEncoder)
    assert graph.encoder.uses_edge_type is True
    assert causal.encoder.uses_edge_type is True


def test_causality_max_incident_edge_node_residual_selection():
    model = CausalityExpert(
        node_feat_dim=7,
        edge_type_dim=4,
        hidden_dim=8,
        expert_dim=2,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        multilabel=True,
    )
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    edge_residual = torch.tensor([[1.0, 0.0], [2.0, 2.0], [3.0, 0.0]])
    edge_loss = torch.tensor([0.1, 0.9, 0.2])
    residual, score = model._max_incident_node_residual(4, edges, edge_residual, edge_loss, torch.device("cpu"))
    assert torch.allclose(score, torch.tensor([0.1, 0.9, 0.9, 0.2]))
    assert torch.allclose(residual[1], edge_residual[1])
    assert torch.allclose(residual[2], edge_residual[1])
