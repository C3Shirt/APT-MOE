import torch

from apt_moe.models import AttributeExpert, EdgeTypeExpert, NodeTypeExpert, SafeGraphEncoder


def test_node_type_mask_removes_target_type_feature():
    model = NodeTypeExpert(num_node_types=3, hidden_dim=8, num_layers=1, dropout=0.0)
    x = torch.eye(3)
    mask_nodes = torch.tensor([1])
    masked = model.masked_input(x, mask_nodes)
    assert torch.allclose(masked[1], model.mask_token.squeeze(0))
    assert torch.allclose(masked[0], x[0])
    assert torch.allclose(masked[2], x[2])


def test_attribute_mask_keeps_type_and_removes_target_attr():
    model = AttributeExpert(node_type_dim=3, attr_dim=2, hidden_dim=8, num_layers=1, dropout=0.0)
    node_type = torch.eye(3)
    attrs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    masked = model.masked_input(node_type, attrs, torch.tensor([2]))
    assert torch.allclose(masked[2, :3], node_type[2])
    assert torch.allclose(masked[2, 3:], model.mask_token.squeeze(0))
    assert torch.allclose(masked[1, 3:], attrs[1])


def test_edge_expert_does_not_accept_or_use_edge_type_as_input():
    model = EdgeTypeExpert(node_feat_dim=7, edge_type_dim=4, hidden_dim=8, num_layers=1, dropout=0.0)
    assert model.uses_edge_type is False
    assert isinstance(model.encoder, SafeGraphEncoder)
    assert model.edge_scores.__code__.co_argcount == 5
