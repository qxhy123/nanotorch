"""API routes for layer computation visualization."""

from fastapi import APIRouter, HTTPException, status
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.models.schemas import TransformerConfig, TensorData
from app.core.layer_wrapper import (
    EncoderLayerWrapper,
    DecoderLayerWrapper,
    create_encoder_layer,
    create_decoder_layer,
    NANOTORCH_AVAILABLE,
)

router = APIRouter(prefix="/api/v1/layer", tags=["layer"])


# ============================================================================
# Request/Response Models
# ============================================================================

class LayerInput(BaseModel):
    """Input data for layer computation."""
    data: List[List[float]] = Field(..., description="Input tensor data (seq_len, d_model)")
    shape: List[int] = Field(..., description="Shape of the input tensor")
    dtype: str = Field("float32", description="Data type")


class EncoderLayerInput(BaseModel):
    """Input for encoder layer computation."""
    config: TransformerConfig = Field(..., description="Transformer configuration")
    input_data: LayerInput = Field(..., description="Layer input data")
    src_mask: Optional[List[List[float]]] = Field(None, description="Source attention mask")
    is_causal: bool = Field(False, description="Whether to use causal attention")


class DecoderLayerInput(BaseModel):
    """Input for decoder layer computation."""
    config: TransformerConfig = Field(..., description="Transformer configuration")
    input_data: LayerInput = Field(..., description="Decoder input data (target)")
    encoder_output: LayerInput = Field(..., description="Encoder output (memory)")
    tgt_mask: Optional[List[List[float]]] = Field(None, description="Target attention mask")
    memory_mask: Optional[List[List[float]]] = Field(None, description="Memory attention mask")


class LayerComputeOptions(BaseModel):
    """Options for layer computation."""
    return_attention_weights: bool = Field(True, description="Return attention weights")
    return_parameters: bool = Field(False, description="Return layer parameters")
    batch_size: int = Field(1, ge=1, le=8, description="Batch size for computation")


class SublayerComputation(BaseModel):
    """Results from a single sublayer computation."""
    norm_input: Optional[TensorData] = Field(None, description="Input to layer norm")
    norm_output: Optional[TensorData] = Field(None, description="Output from layer norm")
    attention: Optional[Dict[str, Any]] = Field(None, description="Attention computation results")
    linear1_output: Optional[TensorData] = Field(None, description="Output from first linear layer")
    activation_output: Optional[TensorData] = Field(None, description="Output after activation")
    linear2_output: Optional[TensorData] = Field(None, description="Output from second linear layer")
    dropout_output: Optional[TensorData] = Field(None, description="Output after dropout")
    residual_output: TensorData = Field(..., description="Output after residual connection")


class EncoderLayerResult(BaseModel):
    """Results from encoder layer computation."""
    success: bool = Field(..., description="Whether computation was successful")
    input: TensorData = Field(..., description="Layer input")
    config: Dict[str, Any] = Field(..., description="Layer configuration")
    sublayer1: SublayerComputation = Field(..., description="Self-attention sublayer results")
    sublayer2: SublayerComputation = Field(..., description="Feed-forward sublayer results")
    output: TensorData = Field(..., description="Layer output")
    error: Optional[str] = Field(None, description="Error message if failed")


class DecoderLayerResult(BaseModel):
    """Results from decoder layer computation."""
    success: bool = Field(..., description="Whether computation was successful")
    input: TensorData = Field(..., description="Layer input (target)")
    encoder_output: TensorData = Field(..., description="Encoder output (memory)")
    config: Dict[str, Any] = Field(..., description="Layer configuration")
    sublayer1: SublayerComputation = Field(..., description="Masked self-attention sublayer results")
    sublayer2: SublayerComputation = Field(..., description="Cross-attention sublayer results")
    sublayer3: SublayerComputation = Field(..., description="Feed-forward sublayer results")
    output: TensorData = Field(..., description="Layer output")
    error: Optional[str] = Field(None, description="Error message if failed")


class LayerStatistics(BaseModel):
    """Statistics about layer computation."""
    num_parameters: int = Field(..., description="Total number of parameters")
    flops: int = Field(..., description="Approximate FLOPs for forward pass")
    memory_mb: float = Field(..., description="Approximate memory usage in MB")
    sublayer_breakdown: Dict[str, int] = Field(..., description="Parameters per sublayer")


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_layer_stats(config: TransformerConfig, layer_type: str = "encoder") -> LayerStatistics:
    """Calculate statistics for a layer.

    Args:
        config: Transformer configuration
        layer_type: Type of layer ("encoder" or "decoder")

    Returns:
        Layer statistics
    """
    d_model = config.d_model
    nhead = config.nhead
    dim_feedforward = config.dim_feedforward

    # Calculate parameters for each component
    # Self-attention: Q, K, V projections + output projection
    attn_params = 4 * d_model * d_model  # 4 weight matrices
    attn_params += 4 * d_model  # 4 bias vectors

    # Feed-forward: 2 linear layers
    ffn_params = d_model * dim_feedforward + dim_feedforward  # Linear1
    ffn_params += dim_feedforward * d_model + d_model  # Linear2

    # Layer norms (2 for encoder, 3 for decoder)
    num_norms = 3 if layer_type == "decoder" else 2
    norm_params = num_norms * (2 * d_model)  # gamma and beta for each norm

    # Total parameters
    total_params = attn_params + ffn_params + norm_params

    # Cross-attention for decoder (same as self-attention)
    if layer_type == "decoder":
        total_params += attn_params

    # Estimate FLOPs (very rough approximation)
    seq_len = 64  # Assume sequence length
    flops = 2 * seq_len * seq_len * d_model  # Self-attention
    flops += 2 * seq_len * d_model * dim_feedforward  # FFN
    flops *= 2  # Multiply by 2 for decoder (cross-attn)

    # Estimate memory
    memory_mb = (total_params * 4) / (1024 * 1024)  # float32 = 4 bytes

    return LayerStatistics(
        num_parameters=total_params,
        flops=flops,
        memory_mb=memory_mb,
        sublayer_breakdown={
            "self_attention": attn_params,
            "feed_forward": ffn_params,
            "layer_norm": norm_params,
        }
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/encoder/compute", response_model=EncoderLayerResult)
async def compute_encoder_layer(request: EncoderLayerInput):
    """Compute encoder layer with intermediate results.

    Args:
        request: Encoder layer computation request

    Returns:
        Detailed computation results with all intermediate tensors
    """
    if not NANOTORCH_AVAILABLE:
        return EncoderLayerResult(
            success=False,
            input=TensorData(shape=[0], data=[], dtype="float32"),
            config={},
            sublayer1=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            sublayer2=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            output=TensorData(shape=[0], data=[], dtype="float32"),
            error="nanotorch is not available"
        )

    try:
        # Create encoder layer wrapper
        layer = create_encoder_layer(request.config.dict())

        # Prepare input
        input_data = request.input_data.data

        # Run forward pass with intermediates
        results = layer.forward_with_intermediates(
            src=input_data,
            src_mask=request.src_mask,
            is_causal=request.is_causal,
        )

        # Add success flag
        results['success'] = True
        results['error'] = None

        return results

    except Exception as e:
        return EncoderLayerResult(
            success=False,
            input=TensorData(
                shape=request.input_data.shape,
                data=request.input_data.data,
                dtype=request.input_data.dtype
            ),
            config=request.config.dict(),
            sublayer1=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            sublayer2=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            output=TensorData(shape=[0], data=[], dtype="float32"),
            error=f"Computation failed: {str(e)}"
        )


@router.post("/decoder/compute", response_model=DecoderLayerResult)
async def compute_decoder_layer(request: DecoderLayerInput):
    """Compute decoder layer with intermediate results.

    Args:
        request: Decoder layer computation request

    Returns:
        Detailed computation results with all intermediate tensors
    """
    if not NANOTORCH_AVAILABLE:
        return DecoderLayerResult(
            success=False,
            input=TensorData(shape=[0], data=[], dtype="float32"),
            encoder_output=TensorData(shape=[0], data=[], dtype="float32"),
            config={},
            sublayer1=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            sublayer2=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            sublayer3=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            output=TensorData(shape=[0], data=[], dtype="float32"),
            error="nanotorch is not available"
        )

    try:
        # Create decoder layer wrapper
        layer = create_decoder_layer(request.config.dict())

        # Prepare inputs
        tgt_data = request.input_data.data
        memory_data = request.encoder_output.data

        # Run forward pass with intermediates
        results = layer.forward_with_intermediates(
            tgt=tgt_data,
            memory=memory_data,
            tgt_mask=request.tgt_mask,
            memory_mask=request.memory_mask,
        )

        # Add success flag
        results['success'] = True
        results['error'] = None

        return results

    except Exception as e:
        return DecoderLayerResult(
            success=False,
            input=TensorData(
                shape=request.input_data.shape,
                data=request.input_data.data,
                dtype=request.input_data.dtype
            ),
            encoder_output=TensorData(
                shape=request.encoder_output.shape,
                data=request.encoder_output.data,
                dtype=request.encoder_output.dtype
            ),
            config=request.config.dict(),
            sublayer1=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            sublayer2=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            sublayer3=SublayerComputation(
                residual_output=TensorData(shape=[0], data=[], dtype="float32")
            ),
            output=TensorData(shape=[0], data=[], dtype="float32"),
            error=f"Computation failed: {str(e)}"
        )


@router.post("/encoder/statistics", response_model=LayerStatistics)
async def get_encoder_statistics(config: TransformerConfig):
    """Get statistics for an encoder layer.

    Args:
        config: Transformer configuration

    Returns:
        Layer statistics including parameters, FLOPs, memory
    """
    return calculate_layer_stats(config, "encoder")


@router.post("/decoder/statistics", response_model=LayerStatistics)
async def get_decoder_statistics(config: TransformerConfig):
    """Get statistics for a decoder layer.

    Args:
        config: Transformer configuration

    Returns:
        Layer statistics including parameters, FLOPs, memory
    """
    stats = calculate_layer_stats(config, "decoder")
    # Add cross-attention parameters for decoder
    d_model = config.d_model
    cross_attn_params = 4 * d_model * d_model + 4 * d_model
    stats.num_parameters += cross_attn_params
    stats.sublayer_breakdown["cross_attention"] = cross_attn_params
    return stats


@router.get("/types")
async def get_layer_types():
    """Get available layer types and their descriptions.

    Returns:
        List of layer types with descriptions
    """
    return {
        "success": True,
        "layer_types": [
            {
                "type": "encoder",
                "name": "Encoder Layer",
                "description": "Transformer encoder layer with self-attention and feed-forward network",
                "sublayers": ["Self-Attention", "Feed-Forward Network"],
                "architecture": "Post-Norm or Pre-Norm"
            },
            {
                "type": "decoder",
                "name": "Decoder Layer",
                "description": "Transformer decoder layer with masked self-attention, cross-attention, and feed-forward network",
                "sublayers": ["Masked Self-Attention", "Cross-Attention", "Feed-Forward Network"],
                "architecture": "Post-Norm or Pre-Norm"
            }
        ]
    }
