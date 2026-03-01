"""Recurrent neural network layers for nanotorch."""

import math
from typing import Optional, Tuple, List
import numpy as np
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
from nanotorch.nn.dropout import Dropout


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

        # Process each layer
        layer_input = x
        for layer in range(self.num_layers):
            # Process forward and backward directions separately for bidirectional RNN
            if self.bidirectional:
                # Forward direction (process sequence in order)
                forward_outputs: List[Tensor] = []
                cell_forward = self._cells[layer * 2]

                for t in range(seq_len):
                    x_t = layer_input[t]
                    if self.mode == "LSTM":
                        h, c = hidden_state
                        h_layer = h[layer * 2]
                        c_layer = c[layer * 2]
                        h_new, c_new = cell_forward(x_t, (h_layer, c_layer))
                        h.data[layer * 2] = h_new.data
                        c.data[layer * 2] = c_new.data
                        hidden_state = (h, c)
                        forward_outputs.append(h_new)
                    else:
                        h = hidden_state
                        h_layer = h[layer * 2]
                        h_new = cell_forward(x_t, h_layer)
                        h.data[layer * 2] = h_new.data
                        hidden_state = h
                        forward_outputs.append(h_new)

                # Backward direction (process sequence in reverse order)
                backward_outputs: List[Tensor] = []
                cell_backward = self._cells[layer * 2 + 1]

                for t in range(seq_len - 1, -1, -1):
                    x_t = layer_input[t]
                    if self.mode == "LSTM":
                        h, c = hidden_state
                        h_layer = h[layer * 2 + 1]
                        c_layer = c[layer * 2 + 1]
                        h_new, c_new = cell_backward(x_t, (h_layer, c_layer))
                        h.data[layer * 2 + 1] = h_new.data
                        c.data[layer * 2 + 1] = c_new.data
                        hidden_state = (h, c)
                        backward_outputs.insert(0, h_new)  # Insert at beginning to reverse order
                    else:
                        h = hidden_state
                        h_layer = h[layer * 2 + 1]
                        h_new = cell_backward(x_t, h_layer)
                        h.data[layer * 2 + 1] = h_new.data
                        hidden_state = h
                        backward_outputs.insert(0, h_new)

                # Concatenate forward and backward outputs
                layer_output_list = []
                for t in range(seq_len):
                    concat = Tensor(
                        np.concatenate([forward_outputs[t].data, backward_outputs[t].data], axis=-1),
                        requires_grad=forward_outputs[t].requires_grad or backward_outputs[t].requires_grad
                    )
                    layer_output_list.append(concat)
            else:
                # Unidirectional RNN
                layer_outputs: List[Tensor] = []
                cell = self._cells[layer]

                for t in range(seq_len):
                    x_t = layer_input[t]
                    if self.mode == "LSTM":
                        h, c = hidden_state
                        h_layer = h[layer]
                        c_layer = c[layer]
                        h_new, c_new = cell(x_t, (h_layer, c_layer))
                        h.data[layer] = h_new.data
                        c.data[layer] = c_new.data
                        hidden_state = (h, c)
                        layer_outputs.append(h_new)
                    else:
                        h = hidden_state
                        h_layer = h[layer]
                        h_new = cell(x_t, h_layer)
                        h.data[layer] = h_new.data
                        hidden_state = h
                        layer_outputs.append(h_new)

                layer_output_list = layer_outputs

            # Apply dropout between layers (not after the last layer)
            if layer < self.num_layers - 1 and self._dropouts[layer] is not None:
                dropout_layer = self._dropouts[layer]
                dropped_outputs = []
                for out in layer_output_list:
                    dropped_outputs.append(dropout_layer(out))
                layer_output_list = dropped_outputs

            # Stack outputs for next layer
            if len(layer_output_list) > 0:
                layer_input = Tensor(
                    np.stack([o.data for o in layer_output_list], axis=0),
                    requires_grad=any(o.requires_grad for o in layer_output_list)
                )

        # Final output
        output = layer_input

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
