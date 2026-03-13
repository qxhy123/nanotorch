import numpy as np

from nanotorch.tensor import Tensor
from nanotorch.nn.recommender import (
    DeepFM,
    WideDeep,
    TwoTowerModel,
    NeuralCF,
    DCN,
    SparseFeat,
    DenseFeat,
)


def _has_nonzero_grad(model) -> bool:
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    return bool(grads) and any(np.any(np.abs(grad.data) > 0) for grad in grads)


def _sparse_features():
    return [
        SparseFeat("user_id", 16, 4),
        SparseFeat("item_id", 32, 4),
    ]


def _dense_features():
    return [DenseFeat("price", 1), DenseFeat("score", 1)]


def test_deepfm_backward_propagates_to_parameters():
    model = DeepFM(_sparse_features(), _dense_features(), embed_dim=4, hidden_dims=[8, 4])
    sparse_input = Tensor(np.array([[1, 3], [2, 5], [4, 7]], dtype=np.float32))
    dense_input = Tensor(np.random.randn(3, 2).astype(np.float32), requires_grad=True)

    model.zero_grad()
    loss = model(sparse_input, dense_input).sum()
    loss.backward()

    assert _has_nonzero_grad(model)


def test_widedeep_backward_propagates_to_parameters():
    model = WideDeep(_sparse_features(), _dense_features(), embed_dim=4, hidden_dims=[8, 4])
    sparse_input = Tensor(np.array([[1, 3], [2, 5]], dtype=np.float32))
    dense_input = Tensor(np.random.randn(2, 2).astype(np.float32), requires_grad=True)

    model.zero_grad()
    loss = model(sparse_input, dense_input).sum()
    loss.backward()

    assert _has_nonzero_grad(model)


def test_two_tower_backward_propagates_to_parameters():
    user_sparse = [SparseFeat("user_id", 16, 4)]
    item_sparse = [SparseFeat("item_id", 32, 4)]
    user_dense = [DenseFeat("age", 1)]
    item_dense = [DenseFeat("price", 1)]
    model = TwoTowerModel(user_sparse, user_dense, item_sparse, item_dense, embed_dim=4, tower_dim=8, hidden_dims=[8])

    user_sparse_input = Tensor(np.array([[1], [2], [3]], dtype=np.float32))
    item_sparse_input = Tensor(np.array([[4], [5], [6]], dtype=np.float32))
    user_dense_input = Tensor(np.random.randn(3, 1).astype(np.float32), requires_grad=True)
    item_dense_input = Tensor(np.random.randn(3, 1).astype(np.float32), requires_grad=True)

    model.zero_grad()
    loss = model(user_sparse_input, item_sparse_input, user_dense_input, item_dense_input).sum()
    loss.backward()

    assert _has_nonzero_grad(model)


def test_neuralcf_backward_propagates_to_parameters():
    model = NeuralCF(num_users=32, num_items=64, embed_dim=4, hidden_dims=[8, 4])
    user_indices = Tensor(np.array([1, 2, 3], dtype=np.float32))
    item_indices = Tensor(np.array([4, 5, 6], dtype=np.float32))

    model.zero_grad()
    loss = model(user_indices, item_indices).sum()
    loss.backward()

    assert _has_nonzero_grad(model)


def test_dcn_backward_propagates_to_parameters():
    model = DCN(_sparse_features(), _dense_features(), embed_dim=4, hidden_dims=[8, 4], num_cross_layers=2)
    sparse_input = Tensor(np.array([[1, 3], [2, 5]], dtype=np.float32))
    dense_input = Tensor(np.random.randn(2, 2).astype(np.float32), requires_grad=True)

    model.zero_grad()
    loss = model(sparse_input, dense_input).sum()
    loss.backward()

    assert _has_nonzero_grad(model)
