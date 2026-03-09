"""Layer wrapper for capturing intermediate computation results."""

import sys
import os
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING

# Add parent directory to path to import nanotorch
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: core/ -> app/ -> backend/ -> nanotorch/
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
sys.path.insert(0, _project_root)

if TYPE_CHECKING:
    from nanotorch.tensor import Tensor

try:
    from nanotorch.nn.transformer import TransformerEncoderLayer, TransformerDecoderLayer
    from nanotorch.nn.linear import Linear
    from nanotorch.nn.dropout import Dropout
    from nanotorch.nn.normalization import LayerNorm
    from nanotorch.nn.activation import ReLU, GELU
    from nanotorch.nn.attention import MultiheadAttention
    from nanotorch.tensor import Tensor
    NANOTORCH_AVAILABLE = True
except ImportError:
    NANOTORCH_AVAILABLE = False
    # Create a dummy type for type checking when nanotorch is not available
    Tensor = Any  # type: ignore


def serialize_tensor(tensor: Union[Tensor, Any]) -> Dict[str, Any]:
    """Convert nanotorch Tensor to serializable format.

    Args:
        tensor: nanotorch Tensor to serialize

    Returns:
        Dictionary with shape, data, and dtype
    """
    # Handle nanotorch Tensor
    if hasattr(tensor, 'data'):
        # Get numpy array from tensor
        arr = tensor.numpy() if hasattr(tensor, 'numpy') else tensor.data
        return {
            "shape": list(arr.shape),
            "data": arr.tolist(),
            "dtype": str(arr.dtype),
        }
    # Handle numpy array directly
    elif hasattr(tensor, 'shape'):
        return {
            "shape": list(tensor.shape),
            "data": tensor.tolist(),
            "dtype": str(tensor.dtype),
        }
    # Handle list
    elif isinstance(tensor, list):
        import numpy as np
        arr = np.array(tensor)
        return {
            "shape": list(arr.shape),
            "data": tensor,
            "dtype": str(arr.dtype),
        }
    else:
        return {
            "shape": [],
            "data": [],
            "dtype": "unknown",
        }


class EncoderLayerWrapper:
    """Wrapper for TransformerEncoderLayer that captures intermediate results.

    This wrapper executes the encoder layer step by step and captures
    all intermediate tensors for visualization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        if not NANOTORCH_AVAILABLE:
            raise RuntimeError("nanotorch is not available")

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.layer_norm_eps = layer_norm_eps

        # Create the actual encoder layer
        self.layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )

        # Access subcomponents for step-by-step execution
        self.self_attn = self.layer.self_attn
        self.linear1 = self.layer.linear1
        self.linear2 = self.layer.linear2
        self.norm1 = self.layer.norm1
        self.norm2 = self.layer.norm2
        self.dropout1 = self.layer.dropout1
        self.dropout2 = self.layer.dropout2
        self.dropout3 = self.layer.dropout3
        self.activation = self.layer.activation

    def forward_with_intermediates(
        self,
        src: Union[Tensor, List[List[float]]],
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass with intermediate results.

        Args:
            src: Input tensor
            src_mask: Attention mask
            src_key_padding_mask: Key padding mask
            is_causal: Whether to use causal attention

        Returns:
            Dictionary containing all intermediate computation results
        """
        results = {}

        # Convert input to tensor if needed
        if not isinstance(src, Tensor):
            import numpy as np
            arr = np.array(src)

            # Handle 2D input: add batch dimension if batch_first=True
            if len(arr.shape) == 2:
                if self.batch_first:
                    # Shape is (seq_len, d_model), need (batch, seq_len, d_model)
                    arr = arr[np.newaxis, :]  # Add batch dimension at front
                # else: keep as (seq_len, d_model) for seq_first

            # Create tensor with proper shape
            src = Tensor(arr)

        # Store input
        results['input'] = serialize_tensor(src)

        # Store configuration
        results['config'] = {
            'd_model': self.d_model,
            'nhead': self.nhead,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'activation': self.activation.__class__.__name__.lower(),
            'norm_first': self.norm_first,
            'layer_norm_eps': self.layer_norm_eps,
            'batch_first': self.batch_first,
        }

        x = src
        results['sublayer1'] = {}
        results['sublayer2'] = {}

        if self.norm_first:
            # ===== Sublayer 1: Self-Attention with Pre-Norm =====

            # Step 1: LayerNorm
            norm1_out = self.norm1(x)
            results['sublayer1']['norm_input'] = serialize_tensor(x)
            results['sublayer1']['norm_output'] = serialize_tensor(norm1_out)

            # Step 2: Self-Attention
            attn_out, attn_weights = self.self_attn(
                norm1_out, norm1_out, norm1_out,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                is_causal=is_causal,
            )
            results['sublayer1']['attention'] = {
                'output': serialize_tensor(attn_out),
            }

            # Step 3: Dropout
            dropout_out = self.dropout1(attn_out)
            results['sublayer1']['dropout_output'] = serialize_tensor(dropout_out)

            # Step 4: Residual Connection
            residual_out = x + dropout_out
            results['sublayer1']['residual_output'] = serialize_tensor(residual_out)

            # Update x for next sublayer
            x = residual_out

            # ===== Sublayer 2: FeedForward with Pre-Norm =====

            # Step 1: LayerNorm
            norm2_out = self.norm2(x)
            results['sublayer2']['norm_input'] = serialize_tensor(x)
            results['sublayer2']['norm_output'] = serialize_tensor(norm2_out)

            # Step 2: Linear1
            linear1_out = self.linear1(norm2_out)
            results['sublayer2']['linear1_output'] = serialize_tensor(linear1_out)

            # Step 3: Activation
            act_out = self.activation(linear1_out)
            results['sublayer2']['activation_output'] = serialize_tensor(act_out)

            # Step 4: Dropout
            dropout_out = self.dropout2(act_out)
            results['sublayer2']['dropout1_output'] = serialize_tensor(dropout_out)

            # Step 5: Linear2
            linear2_out = self.linear2(dropout_out)
            results['sublayer2']['linear2_output'] = serialize_tensor(linear2_out)

            # Step 6: Dropout
            dropout_out = self.dropout3(linear2_out)
            results['sublayer2']['dropout2_output'] = serialize_tensor(dropout_out)

            # Step 7: Residual Connection
            residual_out = x + dropout_out
            results['sublayer2']['residual_output'] = serialize_tensor(residual_out)

            # Final output
            results['output'] = serialize_tensor(residual_out)

        else:
            # ===== Post-Norm Architecture =====

            # ===== Sublayer 1: Self-Attention with Post-Norm =====

            # Step 1: Self-Attention
            attn_out, attn_weights = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                is_causal=is_causal,
            )
            results['sublayer1']['attention'] = {
                'output': serialize_tensor(attn_out),
            }

            # Step 2: Dropout
            dropout_out = self.dropout1(attn_out)
            results['sublayer1']['dropout_output'] = serialize_tensor(dropout_out)

            # Step 3: Residual Connection
            residual_out = x + dropout_out
            results['sublayer1']['residual_output'] = serialize_tensor(residual_out)

            # Step 4: LayerNorm
            norm_out = self.norm1(residual_out)
            results['sublayer1']['norm_input'] = serialize_tensor(residual_out)
            results['sublayer1']['norm_output'] = serialize_tensor(norm_out)

            # Update x for next sublayer
            x = norm_out

            # ===== Sublayer 2: FeedForward with Post-Norm =====

            # Step 1: Linear1
            linear1_out = self.linear1(x)
            results['sublayer2']['linear1_output'] = serialize_tensor(linear1_out)

            # Step 2: Activation
            act_out = self.activation(linear1_out)
            results['sublayer2']['activation_output'] = serialize_tensor(act_out)

            # Step 3: Dropout
            dropout_out = self.dropout2(act_out)
            results['sublayer2']['dropout1_output'] = serialize_tensor(dropout_out)

            # Step 4: Linear2
            linear2_out = self.linear2(dropout_out)
            results['sublayer2']['linear2_output'] = serialize_tensor(linear2_out)

            # Step 5: Dropout
            dropout_out = self.dropout3(linear2_out)
            results['sublayer2']['dropout2_output'] = serialize_tensor(dropout_out)

            # Step 6: Residual Connection
            residual_out = x + dropout_out
            results['sublayer2']['residual_output'] = serialize_tensor(residual_out)

            # Step 7: LayerNorm
            norm_out = self.norm2(residual_out)
            results['sublayer2']['norm_input'] = serialize_tensor(residual_out)
            results['sublayer2']['norm_output'] = serialize_tensor(norm_out)

            # Final output
            results['output'] = serialize_tensor(norm_out)

        return results


class DecoderLayerWrapper:
    """Wrapper for TransformerDecoderLayer that captures intermediate results.

    This wrapper executes the decoder layer step by step and captures
    all intermediate tensors for visualization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        if not NANOTORCH_AVAILABLE:
            raise RuntimeError("nanotorch is not available")

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.layer_norm_eps = layer_norm_eps

        # Create the actual decoder layer
        self.layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )

        # Access subcomponents
        self.self_attn = self.layer.self_attn
        self.multihead_attn = self.layer.multihead_attn
        self.linear1 = self.layer.linear1
        self.linear2 = self.layer.linear2
        self.norm1 = self.layer.norm1
        self.norm2 = self.layer.norm2
        self.norm3 = self.layer.norm3
        self.dropout1 = self.layer.dropout1
        self.dropout2 = self.layer.dropout2
        self.dropout3 = self.layer.dropout3
        self.dropout4 = self.layer.dropout4
        self.dropout5 = self.layer.dropout5
        self.activation = self.layer.activation

    def forward_with_intermediates(
        self,
        tgt: Union[Tensor, List[List[float]]],
        memory: Union[Tensor, List[List[float]]],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with intermediate results.

        Args:
            tgt: Target tensor (decoder input)
            memory: Memory tensor (encoder output)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask

        Returns:
            Dictionary containing all intermediate computation results
        """
        results = {}

        # Convert inputs to tensors if needed
        if not isinstance(tgt, Tensor):
            import numpy as np
            arr = np.array(tgt)

            # Handle 2D input: add batch dimension if batch_first=True
            if len(arr.shape) == 2:
                if self.batch_first:
                    # Shape is (seq_len, d_model), need (batch, seq_len, d_model)
                    arr = arr[np.newaxis, :]  # Add batch dimension at front

            tgt = Tensor(arr)

        if not isinstance(memory, Tensor):
            import numpy as np
            arr = np.array(memory)

            # Handle 2D input: add batch dimension if batch_first=True
            if len(arr.shape) == 2:
                if self.batch_first:
                    # Shape is (seq_len, d_model), need (batch, seq_len, d_model)
                    arr = arr[np.newaxis, :]  # Add batch dimension at front

            memory = Tensor(arr)

        # Store inputs
        results['input'] = serialize_tensor(tgt)
        results['encoder_output'] = serialize_tensor(memory)

        # Store configuration
        results['config'] = {
            'd_model': self.d_model,
            'nhead': self.nhead,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'activation': self.activation.__class__.__name__.lower(),
            'norm_first': self.norm_first,
            'layer_norm_eps': self.layer_norm_eps,
            'batch_first': self.batch_first,
        }

        x = tgt
        results['sublayer1'] = {}  # Masked Self-Attention
        results['sublayer2'] = {}  # Cross-Attention
        results['sublayer3'] = {}  # FeedForward

        if self.norm_first:
            # Pre-Norm architecture
            # (Similar to encoder but with 3 sublayers)
            pass  # Implementation would follow similar pattern
        else:
            # Post-Norm architecture
            # Sublayer 1: Masked Self-Attention
            attn_out, _ = self.self_attn(
                x, x, x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=True,
                is_causal=True,  # Decoder self-attention is causal
            )
            results['sublayer1']['attention'] = {'output': serialize_tensor(attn_out)}

            dropout_out = self.dropout1(attn_out)
            results['sublayer1']['dropout_output'] = serialize_tensor(dropout_out)

            residual_out = x + dropout_out
            results['sublayer1']['residual_output'] = serialize_tensor(residual_out)

            norm_out = self.norm1(residual_out)
            results['sublayer1']['norm_input'] = serialize_tensor(residual_out)
            results['sublayer1']['norm_output'] = serialize_tensor(norm_out)

            x = norm_out

            # Sublayer 2: Cross-Attention
            attn_out, _ = self.multihead_attn(
                x, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
            )
            results['sublayer2']['attention'] = {'output': serialize_tensor(attn_out)}

            dropout_out = self.dropout2(attn_out)
            results['sublayer2']['dropout_output'] = serialize_tensor(dropout_out)

            residual_out = x + dropout_out
            results['sublayer2']['residual_output'] = serialize_tensor(residual_out)

            norm_out = self.norm2(residual_out)
            results['sublayer2']['norm_input'] = serialize_tensor(residual_out)
            results['sublayer2']['norm_output'] = serialize_tensor(norm_out)

            x = norm_out

            # Sublayer 3: FeedForward (same as encoder)
            linear1_out = self.linear1(x)
            results['sublayer3']['linear1_output'] = serialize_tensor(linear1_out)

            act_out = self.activation(linear1_out)
            results['sublayer3']['activation_output'] = serialize_tensor(act_out)

            dropout_out = self.dropout3(act_out)
            results['sublayer3']['dropout1_output'] = serialize_tensor(dropout_out)

            linear2_out = self.linear2(dropout_out)
            results['sublayer3']['linear2_output'] = serialize_tensor(linear2_out)

            dropout_out = self.dropout4(linear2_out)
            results['sublayer3']['dropout2_output'] = serialize_tensor(dropout_out)

            residual_out = x + dropout_out
            results['sublayer3']['residual_output'] = serialize_tensor(residual_out)

            norm_out = self.norm3(residual_out)
            results['sublayer3']['norm_input'] = serialize_tensor(residual_out)
            results['sublayer3']['norm_output'] = serialize_tensor(norm_out)

            results['output'] = serialize_tensor(norm_out)

        return results


def create_encoder_layer(config: Dict[str, Any]) -> EncoderLayerWrapper:
    """Create an EncoderLayerWrapper from config.

    Args:
        config: Configuration dictionary

    Returns:
        EncoderLayerWrapper instance
    """
    return EncoderLayerWrapper(
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        batch_first=config.get('batch_first', True),
        norm_first=config.get('norm_first', False),
    )


def create_decoder_layer(config: Dict[str, Any]) -> DecoderLayerWrapper:
    """Create a DecoderLayerWrapper from config.

    Args:
        config: Configuration dictionary

    Returns:
        DecoderLayerWrapper instance
    """
    return DecoderLayerWrapper(
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        batch_first=config.get('batch_first', True),
        norm_first=config.get('norm_first', False),
    )
