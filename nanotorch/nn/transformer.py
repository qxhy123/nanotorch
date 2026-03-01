"""Transformer modules for nanotorch."""

from typing import Optional, Callable, Union
from nanotorch.nn.module import Module
from nanotorch.nn.linear import Linear
from nanotorch.nn.dropout import Dropout
from nanotorch.nn.normalization import LayerNorm
from nanotorch.nn.activation import ReLU, GELU
from nanotorch.nn.attention import MultiheadAttention
from nanotorch.tensor import Tensor


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attention and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: The number of expected features in the input.
        nhead: The number of heads in the multiheadattention models.
        dim_feedforward: The dimension of the feedforward network model.
            Default: 2048.
        dropout: The dropout value. Default: 0.1.
        activation: The activation function of the intermediate layer, can be
            a string ("relu" or "gelu") or a callable. Default: "relu".
        layer_norm_eps: The eps value in layer normalization components.
            Default: 1e-5.
        batch_first: If True, input is (batch, seq, feature). Default: True.
        norm_first: If True, layer norm is done prior to attention and
            feedforward operations. Default: False.

    Shape:
        - Input: (batch, seq_len, d_model) if batch_first
        - Output: (batch, seq_len, d_model)

    Examples:
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = Tensor.randn((32, 10, 512))
        >>> out = encoder_layer(src)
        >>> out.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation = ReLU()
            elif activation == "gelu":
                self.activation = GELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        else:
            self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass.

        Args:
            src: Input tensor.
            src_mask: Attention mask.
            src_key_padding_mask: Key padding mask.
            is_causal: Whether to use causal attention.

        Returns:
            Output tensor.
        """
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attention, multi-head attention and feedforward network.

    Args:
        d_model: The number of expected features in the input.
        nhead: The number of heads in the multiheadattention models.
        dim_feedforward: The dimension of the feedforward network model.
        dropout: The dropout value.
        activation: The activation function ("relu" or "gelu").
        layer_norm_eps: The eps value in layer normalization components.
        batch_first: If True, input is (batch, seq, feature).
        norm_first: If True, layer norm is done prior to attention.

    Shape:
        - Input: (batch, seq_len, d_model) if batch_first
        - Output: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.dropout5 = Dropout(dropout)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation = ReLU()
            elif activation == "gelu":
                self.activation = GELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        else:
            self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            tgt = tgt + self._sa_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            tgt = self.norm3(tgt + self._ff_block(tgt))
        return tgt

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        x, _ = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        return self.dropout4(x)


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: An instance of TransformerEncoderLayer.
        num_layers: The number of sub-encoder-layers in the encoder.
        norm: The normalization layer to use.
        enable_nested_tensor: Whether to enable nested tensor input.

    Shape:
        - Input: (batch, seq_len, d_model) if batch_first
        - Output: (batch, seq_len, d_model)

    Examples:
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = Tensor.randn((32, 10, 512))
        >>> out = encoder(src)
        >>> out.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
    ) -> None:
        super().__init__()
        self._layers_list: list = []
        for i in range(num_layers):
            layer = encoder_layer if i == 0 else self._clone_layer(encoder_layer)
            self._layers_list.append(layer)
            self.register_module(f"layer_{i}", layer)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def _clone_layer(self, layer: TransformerEncoderLayer) -> TransformerEncoderLayer:
        new_layer = TransformerEncoderLayer(
            d_model=layer.d_model,
            nhead=layer.nhead,
            dim_feedforward=layer.dim_feedforward,
            dropout=layer.dropout,
            activation=layer.activation,
            batch_first=layer.batch_first,
            norm_first=layer.norm_first,
        )
        return new_layer

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src
        for layer in self._layers_list:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: An instance of TransformerDecoderLayer.
        num_layers: The number of sub-decoder-layers in the decoder.
        norm: The normalization layer to use.

    Shape:
        - Input (tgt): (batch, seq_len, d_model) if batch_first
        - Input (memory): (batch, seq_len, d_model) if batch_first
        - Output: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        super().__init__()
        self._layers_list: list = []
        for i in range(num_layers):
            layer = decoder_layer if i == 0 else self._clone_layer(decoder_layer)
            self._layers_list.append(layer)
            self.register_module(f"layer_{i}", layer)
        self.num_layers = num_layers
        self.norm = norm

    def _clone_layer(self, layer: TransformerDecoderLayer) -> TransformerDecoderLayer:
        new_layer = TransformerDecoderLayer(
            d_model=layer.d_model,
            nhead=layer.nhead,
            dim_feedforward=layer.dim_feedforward,
            dropout=layer.dropout,
            activation=layer.activation,
            batch_first=layer.batch_first,
            norm_first=layer.norm_first,
        )
        return new_layer

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        output = tgt
        for layer in self._layers_list:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(Module):
    """A Transformer model.

    Args:
        d_model: The number of expected features in the encoder/decoder inputs.
        nhead: The number of heads in the multiheadattention models.
        num_encoder_layers: The number of sub-encoder-layers in the encoder.
        num_decoder_layers: The number of sub-decoder-layers in the decoder.
        dim_feedforward: The dimension of the feedforward network model.
        dropout: The dropout value.
        activation: The activation function.
        custom_encoder: Custom encoder.
        custom_decoder: Custom decoder.
        layer_norm_eps: The eps value in layer normalization.
        batch_first: If True, input is (batch, seq, feature).
        norm_first: If True, layer norm is done prior to attention.

    Examples:
        >>> transformer = Transformer(nhead=16, num_encoder_layers=12)
        >>> src = Tensor.randn((32, 10, 512))
        >>> tgt = Tensor.randn((32, 20, 512))
        >>> out = transformer(src, tgt)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "relu",
        custom_encoder: Optional[TransformerEncoder] = None,
        custom_decoder: Optional[TransformerDecoder] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first, norm_first
        )
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = custom_encoder or TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first, norm_first
        )
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = custom_decoder or TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output
