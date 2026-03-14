"""API routes for transformer operations."""

import numpy as np
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional

from app.models.schemas import (
    TransformerConfig,
    TransformerInput,
    TransformerForwardOptions,
    TransformerForwardResponse,
    ValidationResponse,
)
from app.core.transformer_wrapper import (
    create_transformer_from_config,
    NANOTORCH_AVAILABLE,
)

router = APIRouter(prefix="/api/v1/transformer", tags=["transformer"])


def _prepare_tokens(
    text: str,
    explicit_tokens: Optional[List[int]],
    vocab_size: int,
    max_seq_len: int,
) -> np.ndarray:
    """Build a token tensor and enforce max sequence length."""
    if explicit_tokens:
        tokens = np.array(explicit_tokens, dtype=np.int64)
    else:
        tokens = np.array(
            [ord(char) % vocab_size for char in text[:max_seq_len]],
            dtype=np.int64,
        )

    if tokens.ndim == 1:
        tokens = tokens[:max_seq_len].reshape(1, -1)
    else:
        tokens = tokens[:, :max_seq_len]

    return tokens


@router.post("/forward", response_model=TransformerForwardResponse)
async def forward_pass(
    config: TransformerConfig,
    input_data: TransformerInput,
    options: TransformerForwardOptions = TransformerForwardOptions(),
):
    """Run a forward pass through the transformer model.

    Args:
        config: Model configuration
        input_data: Input data
        options: Forward pass options

    Returns:
        Transformer forward response with outputs
    """
    if not NANOTORCH_AVAILABLE:
        return TransformerForwardResponse(
            success=False,
            error="nanotorch is not available. Please ensure it is installed.",
        )

    try:
        # Validate configuration
        if config.d_model % config.nhead != 0:
            return TransformerForwardResponse(
                success=False,
                error=f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})",
            )

        # Create model
        model = create_transformer_from_config(config.dict())

        text = input_data.text
        tokens = _prepare_tokens(
            text=text,
            explicit_tokens=input_data.tokens,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )

        # Process target sequence if provided and decoder exists
        target_tokens = None
        if config.num_decoder_layers > 0 and (input_data.target_text or input_data.target_tokens):
            target_text = input_data.target_text or text
            target_tokens = _prepare_tokens(
                text=target_text,
                explicit_tokens=input_data.target_tokens,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )

        # Run forward pass
        result = model.forward(
            tokens,
            target_tokens=target_tokens,
            return_attention=options.return_attention,
            return_all_layers=options.return_all_layers,
            return_embeddings=options.return_embeddings,
        )

        return TransformerForwardResponse(success=True, data=result)

    except Exception as e:
        return TransformerForwardResponse(
            success=False,
            error=f"Forward pass failed: {str(e)}",
        )


@router.post("/attention")
async def get_attention(
    config: TransformerConfig,
    input_data: TransformerInput,
    layer_index: int = 0,
):
    """Get attention weights for a specific layer.

    Args:
        config: Model configuration
        input_data: Input data
        layer_index: Layer index to get attention for

    Returns:
        Attention weights and related data
    """
    if not NANOTORCH_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="nanotorch is not available",
        )

    try:
        model = create_transformer_from_config(config.dict())

        tokens = _prepare_tokens(
            text=input_data.text,
            explicit_tokens=input_data.tokens,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )

        # Get attention
        result = model.get_attention_weights(tokens, layer_index)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get attention weights: {str(e)}",
        )


@router.post("/embeddings")
async def get_embeddings(
    config: TransformerConfig,
    input_data: TransformerInput,
):
    """Get embeddings for the input text.

    Args:
        config: Model configuration
        input_data: Input data

    Returns:
        Token and positional embeddings
    """
    if not NANOTORCH_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="nanotorch is not available",
        )

    try:
        model = create_transformer_from_config(config.dict())

        tokens = _prepare_tokens(
            text=input_data.text,
            explicit_tokens=input_data.tokens,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )

        # Get embeddings
        result = model.get_embeddings(tokens)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embeddings: {str(e)}",
        )


@router.get("/positional-encoding")
async def get_positional_encoding(
    seq_len: int = 128,
    d_model: int = 512,
):
    """Get positional encoding matrix.

    Args:
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        Positional encoding matrix
    """
    try:
        # Generate positional encoding
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2).astype(np.float32) * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return {
            "success": True,
            "data": {
                "shape": [seq_len, d_model],
                "data": pe.tolist(),
                "dtype": "float32",
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate positional encoding: {str(e)}",
        )


@router.post("/validate-config", response_model=ValidationResponse)
async def validate_config(config: TransformerConfig):
    """Validate a transformer configuration.

    Args:
        config: Configuration to validate

    Returns:
        Validation response
    """
    errors = []

    # Check if d_model is divisible by nhead
    if config.d_model % config.nhead != 0:
        errors.append(
            f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})"
        )

    # Check sequence length
    if config.max_seq_len < 1:
        errors.append("max_seq_len must be at least 1")

    # Check vocabulary size
    if config.vocab_size < 1:
        errors.append("vocab_size must be at least 1")

    # Check layer counts
    if config.num_encoder_layers < 0:
        errors.append("num_encoder_layers cannot be negative")

    if config.num_decoder_layers < 0:
        errors.append("num_decoder_layers cannot be negative")

    if config.num_encoder_layers == 0 and config.num_decoder_layers == 0:
        errors.append("At least one encoder or decoder layer is required")

    return ValidationResponse(valid=len(errors) == 0, errors=errors if errors else None)
