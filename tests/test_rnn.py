import numpy as np

from nanotorch.nn.rnn import GRU, LSTM, RNN
from nanotorch.tensor import Tensor


def _assert_grad_flow(model, x, loss):
    loss.backward()
    assert x.grad is not None
    params_with_grad = 0
    for param in model.parameters():
        if param.grad is not None:
            params_with_grad += 1
    assert params_with_grad > 0


def test_rnn_backward_flow():
    model = RNN(4, 3, num_layers=2, bidirectional=True)
    x = Tensor(np.random.randn(5, 2, 4).astype(np.float32), requires_grad=True)
    output, hidden = model(x)
    loss = output.sum() + hidden.sum()
    _assert_grad_flow(model, x, loss)


def test_lstm_backward_flow():
    model = LSTM(4, 3, num_layers=2, bidirectional=True)
    x = Tensor(np.random.randn(5, 2, 4).astype(np.float32), requires_grad=True)
    output, (h_n, c_n) = model(x)
    loss = output.sum() + h_n.sum() + c_n.sum()
    _assert_grad_flow(model, x, loss)


def test_gru_backward_flow_batch_first():
    model = GRU(4, 3, num_layers=2, batch_first=True)
    x = Tensor(np.random.randn(2, 5, 4).astype(np.float32), requires_grad=True)
    output, hidden = model(x)
    loss = output.sum() + hidden.sum()
    _assert_grad_flow(model, x, loss)
