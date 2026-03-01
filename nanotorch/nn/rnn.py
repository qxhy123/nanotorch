"""Recurrent neural network layers for nanotorch."""

import math
from typing import Optional, Tuple, List
import numpy as np
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor


class RNNCell(Module):
    """An Elman RNN cell with tanh or ReLU non-linearity.

    h' = tanh(W_{ih} x + b_{ih} + W_{hh} h + b_{hh})

    Args:
        input_size: Number of expected features in input x.
        hidden_size: Number of features in the hidden state h.
        bias: If False, no bias is added. Default: True.
        nonlinearity: The non-linearity to use. Can be 'tanh' or 'relu'. Default: 'tanh'.

    Shape:
        - Input: (batch, input_size)
        - Hidden: (batch, hidden_size)
        - Output: (batch, hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        self.weight_ih = Tensor(
            np.random.randn(input_size, hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )

        self.bias_ih: Optional[Tensor] = None
        self.bias_hh: Optional[Tensor] = None

        if bias:
            self.bias_ih = Tensor(
                np.zeros(hidden_size, dtype=np.float32), requires_grad=True
            )
            self.bias_hh = Tensor(
                np.zeros(hidden_size, dtype=np.float32), requires_grad=True
            )

        self.register_parameter("weight_ih", self.weight_ih)
        self.register_parameter("weight_hh", self.weight_hh)
        if bias:
            self.register_parameter("bias_ih", self.bias_ih)
            self.register_parameter("bias_hh", self.bias_hh)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = Tensor(
                np.zeros((x.shape[0], self.hidden_size), dtype=np.float32),
                requires_grad=x.requires_grad,
            )

        ih = x.matmul(self.weight_ih)
        if self.bias_ih is not None:
            ih = ih + self.bias_ih

        hh = h.matmul(self.weight_hh)
        if self.bias_hh is not None:
            hh = hh + self.bias_hh

        out = ih + hh

        if self.nonlinearity == "tanh":
            out = out.tanh()
        elif self.nonlinearity == "relu":
            out = out.relu()
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")

        return out


class LSTMCell(Module):
    """A long short-term memory (LSTM) cell.

    Args:
        input_size: Number of expected features in input x.
        hidden_size: Number of features in the hidden state h.
        bias: If False, no bias is added. Default: True.

    Shape:
        - Input: (batch, input_size)
        - Hidden: (batch, hidden_size), (batch, hidden_size)  # h, c
        - Output: (batch, hidden_size), (batch, hidden_size)  # h', c'
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_size = 4 * hidden_size

        self.weight_ih = Tensor(
            np.random.randn(input_size, gate_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, gate_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )

        self.bias_ih: Optional[Tensor] = None
        self.bias_hh: Optional[Tensor] = None

        if bias:
            self.bias_ih = Tensor(
                np.zeros(gate_size, dtype=np.float32), requires_grad=True
            )
            self.bias_hh = Tensor(
                np.zeros(gate_size, dtype=np.float32), requires_grad=True
            )

        self.register_parameter("weight_ih", self.weight_ih)
        self.register_parameter("weight_hh", self.weight_hh)
        if bias:
            self.register_parameter("bias_ih", self.bias_ih)
            self.register_parameter("bias_hh", self.bias_hh)

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        if state is None:
            h = Tensor(
                np.zeros((x.shape[0], self.hidden_size), dtype=np.float32),
                requires_grad=x.requires_grad,
            )
            c = Tensor(
                np.zeros((x.shape[0], self.hidden_size), dtype=np.float32),
                requires_grad=x.requires_grad,
            )
        else:
            h, c = state

        gates = x.matmul(self.weight_ih) + h.matmul(self.weight_hh)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        i = gates[:, : self.hidden_size].sigmoid()
        f = gates[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = gates[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = gates[:, 3 * self.hidden_size :].sigmoid()

        c_new = f * c + i * g
        h_new = o * c_new.tanh()

        return h_new, c_new


class GRUCell(Module):
    """A gated recurrent unit (GRU) cell.

    Args:
        input_size: Number of expected features in input x.
        hidden_size: Number of features in the hidden state h.
        bias: If False, no bias is added. Default: True.

    Shape:
        - Input: (batch, input_size)
        - Hidden: (batch, hidden_size)
        - Output: (batch, hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Tensor(
            np.random.randn(input_size, 3 * hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32)
            * math.sqrt(1.0 / hidden_size),
            requires_grad=True,
        )

        self.bias_ih: Optional[Tensor] = None
        self.bias_hh: Optional[Tensor] = None

        if bias:
            self.bias_ih = Tensor(
                np.zeros(3 * hidden_size, dtype=np.float32), requires_grad=True
            )
            self.bias_hh = Tensor(
                np.zeros(3 * hidden_size, dtype=np.float32), requires_grad=True
            )

        self.register_parameter("weight_ih", self.weight_ih)
        self.register_parameter("weight_hh", self.weight_hh)
        if bias:
            self.register_parameter("bias_ih", self.bias_ih)
            self.register_parameter("bias_hh", self.bias_hh)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = Tensor(
                np.zeros((x.shape[0], self.hidden_size), dtype=np.float32),
                requires_grad=x.requires_grad,
            )

        gi = x.matmul(self.weight_ih)
        gh = h.matmul(self.weight_hh)
        if self.bias_ih is not None:
            gi = gi + self.bias_ih
        if self.bias_hh is not None:
            gh = gh + self.bias_hh

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = (i_r + h_r).sigmoid()
        z = (i_z + h_z).sigmoid()
        n = (i_n + r * h_n).tanh()

        h_new = (1 - z) * n + z * h
        return h_new


class RNNBase(Module):
    """Base class for RNN modules.

    Args:
        mode: 'RNN', 'LSTM', or 'GRU'.
        input_size: Number of expected features in input.
        hidden_size: Number of features in hidden state.
        num_layers: Number of recurrent layers.
        bias: If False, no bias is added.
        batch_first: If True, input is (batch, seq, feature).
        dropout: Dropout probability (applied between layers).
        bidirectional: If True, bidirectional RNN.
    """

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

        self.num_directions = 2 if bidirectional else 1

        self._cells: List[Module] = []

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

                if mode == "RNN":
                    cell = RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
                elif mode == "LSTM":
                    cell = LSTMCell(layer_input_size, hidden_size, bias)
                elif mode == "GRU":
                    cell = GRUCell(layer_input_size, hidden_size, bias)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                self._cells.append(cell)
                self.register_module(f"cell_{layer}_{direction}", cell)

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if h_0 is None:
            if self.mode == "LSTM":
                h = Tensor(
                    np.zeros(
                        (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                        dtype=np.float32,
                    ),
                    requires_grad=x.requires_grad,
                )
                c = Tensor(
                    np.zeros(
                        (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                        dtype=np.float32,
                    ),
                    requires_grad=x.requires_grad,
                )
                hidden_state = (h, c)
            else:
                hidden_state = Tensor(
                    np.zeros(
                        (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                        dtype=np.float32,
                    ),
                    requires_grad=x.requires_grad,
                )
        else:
            hidden_state = h_0

        outputs: List[Tensor] = []

        for t in range(seq_len):
            x_t = x[t]
            layer_outputs: List[Tensor] = []

            for layer in range(self.num_layers):
                for direction in range(self.num_directions):
                    cell_idx = layer * self.num_directions + direction
                    cell = self._cells[cell_idx]

                    if self.mode == "LSTM":
                        h, c = hidden_state
                        h_layer = h[cell_idx]
                        c_layer = c[cell_idx]
                        h_new, c_new = cell(x_t, (h_layer, c_layer))
                        h.data[cell_idx] = h_new.data
                        c.data[cell_idx] = c_new.data
                        layer_outputs.append(h_new)
                        hidden_state = (h, c)
                    else:
                        h = hidden_state
                        h_layer = h[cell_idx]
                        h_new = cell(x_t, h_layer)
                        h.data[cell_idx] = h_new.data
                        layer_outputs.append(h_new)
                        hidden_state = h

                if self.num_directions == 2:
                    x_t = Tensor(
                        np.concatenate([layer_outputs[0].data, layer_outputs[1].data], axis=-1)
                    )
                else:
                    x_t = layer_outputs[0]

            outputs.append(x_t)

        output = Tensor(
            np.stack([o.data for o in outputs], axis=0)
        )

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden_state


class RNN(RNNBase):
    """Elman RNN with tanh or ReLU non-linearity.

    Args:
        input_size: Number of expected features in input.
        hidden_size: Number of features in hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        nonlinearity: 'tanh' or 'relu'. Default: 'tanh'.
        bias: If False, no bias. Default: True.
        batch_first: If True, input is (batch, seq, feature). Default: False.
        dropout: Dropout probability. Default: 0.
        bidirectional: If True, bidirectional RNN. Default: False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            mode="RNN",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )


class LSTM(RNNBase):
    """Long Short-Term Memory network.

    Args:
        input_size: Number of expected features in input.
        hidden_size: Number of features in hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        bias: If False, no bias. Default: True.
        batch_first: If True, input is (batch, seq, feature). Default: False.
        dropout: Dropout probability. Default: 0.
        bidirectional: If True, bidirectional LSTM. Default: False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            mode="LSTM",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )


class GRU(RNNBase):
    """Gated Recurrent Unit network.

    Args:
        input_size: Number of expected features in input.
        hidden_size: Number of features in hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        bias: If False, no bias. Default: True.
        batch_first: If True, input is (batch, seq, feature). Default: False.
        dropout: Dropout probability. Default: 0.
        bidirectional: If True, bidirectional GRU. Default: False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            mode="GRU",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
