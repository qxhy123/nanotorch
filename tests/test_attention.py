import numpy as np

from nanotorch.nn.attention import MultiheadAttention
from nanotorch.tensor import Tensor


def test_multihead_attention_backward_with_key_padding_mask():
    attn = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
    query = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    key = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    value = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    key_padding_mask = Tensor(np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float32))

    output, weights = attn(query, key, value, key_padding_mask=key_padding_mask)
    loss = output.sum() + weights.sum()
    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    grad_params = sum(1 for param in attn.parameters() if param.grad is not None)
    assert grad_params > 0
