import numpy as np

from nanotorch.nn.embedding import Embedding, EmbeddingBag
from nanotorch.tensor import Tensor


def test_embedding_backward_accumulates_repeated_indices():
    embedding = Embedding(6, 3, padding_idx=0)
    embedding.weight.data = np.arange(18, dtype=np.float32).reshape(6, 3)

    indices = Tensor(np.array([[1, 2], [1, 0]], dtype=np.int64))
    output = embedding(indices)
    loss = output.sum()
    loss.backward()

    assert embedding.weight.grad is not None
    expected = np.zeros_like(embedding.weight.data)
    expected[1] = 2.0
    expected[2] = 1.0
    assert np.allclose(embedding.weight.grad.data, expected)


def test_embedding_bag_backward_sum_and_max():
    bag_sum = EmbeddingBag(5, 2, mode="sum", padding_idx=0)
    bag_sum.weight.data = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    indices = Tensor(np.array([[1, 2], [2, 0]], dtype=np.int64))
    output = bag_sum(indices)
    output.sum().backward()

    assert bag_sum.weight.grad is not None
    expected_sum = np.zeros_like(bag_sum.weight.data)
    expected_sum[1] = 1.0
    expected_sum[2] = 2.0
    assert np.allclose(bag_sum.weight.grad.data, expected_sum)

    bag_max = EmbeddingBag(5, 2, mode="max")
    bag_max.weight.data = np.array(
        [
            [0.0, 0.0],
            [1.0, 5.0],
            [4.0, 2.0],
            [3.0, 6.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int64))
    output = bag_max(indices)
    output.sum().backward()

    assert bag_max.weight.grad is not None
    expected_max = np.zeros_like(bag_max.weight.data)
    expected_max[1, 1] = 1.0
    expected_max[2, 0] = 1.0
    expected_max[3] = 1.0
    expected_max[4, 0] = 0.0
    assert np.allclose(bag_max.weight.grad.data, expected_max)
