"""Recurrent neural network layers for nanotorch."""

import math
from typing import Optional, Tuple, List
import numpy as np
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
from nanotorch.nn.dropout import Dropout
from nanotorch.utils import cat, stack


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
            # Initialize forget gate bias to 1.0 for better memory retention
            # Gates order: input, forget, cell, output
            bias_init = np.zeros(gate_size, dtype=np.float32)
            bias_init[hidden_size:2*hidden_size] = 1.0  # Forget gate bias
            self.bias_ih = Tensor(bias_init.copy(), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(gate_size, dtype=np.float32), requires_grad=True)

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

        # Create dropout layers for inter-layer dropout (only if dropout > 0 and num_layers > 1)
        self._dropouts: List[Optional[Dropout]] = []
        if dropout > 0 and num_layers > 1:
            for layer in range(num_layers - 1):
                dropout_layer = Dropout(dropout)
                self._dropouts.append(dropout_layer)
                self.register_module(f"dropout_{layer}", dropout_layer)
        else:
            self._dropouts = [None] * (num_layers - 1) if num_layers > 1 else []

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        total_states = self.num_layers * self.num_directions

        if self.mode == "LSTM":
            if h_0 is None:
                h_states = [
                    Tensor(
                        np.zeros((batch_size, self.hidden_size), dtype=np.float32),
                        requires_grad=x.requires_grad,
                    )
                    for _ in range(total_states)
                ]
                c_states = [
                    Tensor(
                        np.zeros((batch_size, self.hidden_size), dtype=np.float32),
                        requires_grad=x.requires_grad,
                    )
                    for _ in range(total_states)
                ]
            else:
                h_init, c_init = h_0
                h_states = [h_init[i] for i in range(total_states)]
                c_states = [c_init[i] for i in range(total_states)]
        else:
            if h_0 is None:
                h_states = [
                    Tensor(
                        np.zeros((batch_size, self.hidden_size), dtype=np.float32),
                        requires_grad=x.requires_grad,
                    )
                    for _ in range(total_states)
                ]
            else:
                h_states = [h_0[i] for i in range(total_states)]

        layer_input = x
        for layer in range(self.num_layers):
            if self.bidirectional:
                forward_index = layer * 2
                backward_index = forward_index + 1
                cell_forward = self._cells[forward_index]
                cell_backward = self._cells[backward_index]
                forward_outputs: List[Tensor] = []
                backward_outputs: List[Tensor] = []

                if self.mode == "LSTM":
                    forward_h = h_states[forward_index]
                    forward_c = c_states[forward_index]
                    backward_h = h_states[backward_index]
                    backward_c = c_states[backward_index]

                    for t in range(seq_len):
                        forward_h, forward_c = cell_forward(layer_input[t], (forward_h, forward_c))
                        forward_outputs.append(forward_h)

                    for t in range(seq_len - 1, -1, -1):
                        backward_h, backward_c = cell_backward(layer_input[t], (backward_h, backward_c))
                        backward_outputs.insert(0, backward_h)

                    h_states[forward_index] = forward_h
                    c_states[forward_index] = forward_c
                    h_states[backward_index] = backward_h
                    c_states[backward_index] = backward_c
                else:
                    forward_h = h_states[forward_index]
                    backward_h = h_states[backward_index]

                    for t in range(seq_len):
                        forward_h = cell_forward(layer_input[t], forward_h)
                        forward_outputs.append(forward_h)

                    for t in range(seq_len - 1, -1, -1):
                        backward_h = cell_backward(layer_input[t], backward_h)
                        backward_outputs.insert(0, backward_h)

                    h_states[forward_index] = forward_h
                    h_states[backward_index] = backward_h

                layer_output_list = [
                    cat([forward_outputs[t], backward_outputs[t]], dim=-1)
                    for t in range(seq_len)
                ]
            else:
                state_index = layer
                cell = self._cells[layer]
                layer_output_list: List[Tensor] = []

                if self.mode == "LSTM":
                    current_h = h_states[state_index]
                    current_c = c_states[state_index]
                    for t in range(seq_len):
                        current_h, current_c = cell(layer_input[t], (current_h, current_c))
                        layer_output_list.append(current_h)
                    h_states[state_index] = current_h
                    c_states[state_index] = current_c
                else:
                    current_h = h_states[state_index]
                    for t in range(seq_len):
                        current_h = cell(layer_input[t], current_h)
                        layer_output_list.append(current_h)
                    h_states[state_index] = current_h

            if layer < self.num_layers - 1 and self._dropouts[layer] is not None:
                dropout_layer = self._dropouts[layer]
                layer_output_list = [dropout_layer(out) for out in layer_output_list]

            if layer_output_list:
                layer_input = stack(layer_output_list, dim=0)

        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)

        if self.mode == "LSTM":
            hidden_state = (stack(h_states, dim=0), stack(c_states, dim=0))
        else:
            hidden_state = stack(h_states, dim=0)

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
