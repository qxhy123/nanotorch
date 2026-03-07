"""Wrapper class for nanotorch Transformer model."""

import sys
import os
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

# Add parent directory to path to import nanotorch
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
sys.path.insert(0, _project_root)

try:
    from nanotorch.nn.transformer import Transformer
    from nanotorch.nn.embedding import Embedding
    from nanotorch.nn.attention import MultiheadAttention
    from nanotorch.nn.module import Module
    NANOTORCH_AVAILABLE = True
except ImportError:
    NANOTORCH_AVAILABLE = False
    Transformer = None
    Embedding = None
    MultiheadAttention = None
    Module = None


class TransformerWrapper:
    """Wrapper for running Transformer inference with nanotorch."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        vocab_size: int = 10000,
    ):
        """Initialize the Transformer wrapper.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function
            layer_norm_eps: Layer normalization epsilon
            batch_first: Batch first format
            norm_first: Pre-layer normalization
            vocab_size: Vocabulary size
        """
        if not NANOTORCH_AVAILABLE:
            raise RuntimeError("nanotorch is not available. Please ensure it is installed.")

        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
            "norm_first": norm_first,
            "vocab_size": vocab_size,
        }

        # Create embedding layer
        self.embedding = Embedding(vocab_size, d_model)

        # Create transformer
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )

        # Set to evaluation mode
        self.transformer.eval()

    def generate_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Generate sinusoidal positional encoding.

        Args:
            seq_len: Sequence length
            d_model: Model dimension

        Returns:
            Positional encoding array of shape (seq_len, d_model)
        """
        pe = np.zeros((seq_len, d_model), dtype=np.float32)

        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2).astype(np.float32) * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(
        self,
        tokens: np.ndarray,
        target_tokens: Optional[np.ndarray] = None,
        return_attention: bool = True,
        return_all_layers: bool = True,
        return_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Run forward pass through the transformer.

        Args:
            tokens: Source token IDs of shape (batch_size, src_seq_len)
            target_tokens: Target token IDs of shape (batch_size, tgt_seq_len) (optional)
            return_attention: Whether to return attention weights
            return_all_layers: Whether to return all layer outputs
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with outputs
        """
        from nanotorch.tensor import Tensor

        batch_size, seq_len = tokens.shape
        d_model = self.config["d_model"]

        # Process source sequence
        # Get token embeddings
        token_emb = self.embedding(Tensor(tokens))
        token_emb_data = token_emb.data

        # Get positional encodings
        pos_enc = self.generate_positional_encoding(seq_len, d_model)
        pos_enc = pos_enc.reshape(1, seq_len, d_model)
        pos_enc = np.repeat(pos_enc, batch_size, axis=0)

        # Combine embeddings
        src_embeddings = token_emb_data + pos_enc

        results = {}

        if return_embeddings:
            results["embeddings"] = {
                "token_embeddings": {
                    "shape": list(token_emb_data.shape),
                    "data": token_emb_data.tolist(),
                    "dtype": "float32",
                },
                "positional_encodings": {
                    "shape": list(pos_enc.shape),
                    "data": pos_enc.tolist(),
                    "dtype": "float32",
                },
                "combined": {
                    "shape": list(src_embeddings.shape),
                    "data": src_embeddings.tolist(),
                    "dtype": "float32",
                },
            }

        # Run transformer
        src = Tensor(src_embeddings)

        # For decoder-only or encoder-decoder
        if self.config["num_decoder_layers"] > 0:
            # Process target sequence if provided
            if target_tokens is not None:
                tgt_batch_size, tgt_seq_len = target_tokens.shape

                # Get target token embeddings
                tgt_token_emb = self.embedding(Tensor(target_tokens))
                tgt_token_emb_data = tgt_token_emb.data

                # Get target positional encodings
                tgt_pos_enc = self.generate_positional_encoding(tgt_seq_len, d_model)
                tgt_pos_enc = tgt_pos_enc.reshape(1, tgt_seq_len, d_model)
                tgt_pos_enc = np.repeat(tgt_pos_enc, tgt_batch_size, axis=0)

                # Combine target embeddings
                tgt_embeddings = tgt_token_emb_data + tgt_pos_enc
                tgt = Tensor(tgt_embeddings)

                # Store target embeddings in results
                if return_embeddings:
                    results["target_embeddings"] = {
                        "token_embeddings": {
                            "shape": list(tgt_token_emb_data.shape),
                            "data": tgt_token_emb_data.tolist(),
                            "dtype": "float32",
                        },
                        "positional_encodings": {
                            "shape": list(tgt_pos_enc.shape),
                            "data": tgt_pos_enc.tolist(),
                            "dtype": "float32",
                        },
                        "combined": {
                            "shape": list(tgt_embeddings.shape),
                            "data": tgt_embeddings.tolist(),
                            "dtype": "float32",
                        },
                    }
            else:
                # No target provided, use source as target (autoencoder-like behavior)
                tgt = Tensor(src_embeddings)

            output = self.transformer(src, tgt)
        else:
            # Encoder only
            output = self.transformer.encoder(src)

        results["final_output"] = {
            "shape": list(output.data.shape),
            "data": output.data.tolist(),
            "dtype": "float32",
        }

        # Note: In a full implementation, we would hook into the layers
        # to collect intermediate outputs and attention weights.
        # For now, we return a placeholder structure.

        if return_all_layers:
            results["layer_outputs"] = []
            # Add placeholder layer outputs
            num_layers = self.config["num_encoder_layers"] + self.config["num_decoder_layers"]
            for i in range(num_layers):
                layer_type = "encoder" if i < self.config["num_encoder_layers"] else "decoder"
                results["layer_outputs"].append({
                    "layer_name": f"{layer_type}_layer_{i}",
                    "layer_type": layer_type,
                    "input_shape": list(src_embeddings.shape),
                    "output_shape": list(src_embeddings.shape),
                    "output": {
                        "shape": list(src_embeddings.shape),
                        "data": src_embeddings.tolist(),
                        "dtype": "float32",
                    },
                })

        if return_attention:
            # Create attention data with Q, K, V projections
            results["attention_weights"] = []
            num_layers = self.config["num_encoder_layers"]
            nhead = self.config["nhead"]
            head_dim = d_model // nhead

            for i in range(num_layers):
                # Simulate Q, K, V projections (for visualization)
                # In a real implementation, these would come from the actual attention layers
                q = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(np.float32) * 0.1
                k = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(np.float32) * 0.1
                v = np.random.randn(batch_size, nhead, seq_len, head_dim).astype(np.float32) * 0.1

                # Compute attention weights from Q and K
                scores = np.matmul(q, k.transpose(0, 1, 3, 2))  # (batch, heads, seq_len, seq_len)
                scores = scores / np.sqrt(head_dim)  # Scale
                attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)  # Normalize

                results["attention_weights"].append({
                    "weights": {
                        "shape": list(attn_weights.shape),
                        "data": attn_weights.tolist(),
                        "dtype": "float32",
                    },
                    "queries": {
                        "shape": list(q.shape),
                        "data": q.tolist(),
                        "dtype": "float32",
                    },
                    "keys": {
                        "shape": list(k.shape),
                        "data": k.tolist(),
                        "dtype": "float32",
                    },
                    "values": {
                        "shape": list(v.shape),
                        "data": v.tolist(),
                        "dtype": "float32",
                    },
                    "scale": 1.0 / np.sqrt(head_dim),
                })

        return results

    def get_attention_weights(
        self,
        tokens: np.ndarray,
        layer_index: int = 0,
    ) -> Dict[str, Any]:
        """Get attention weights for a specific layer.

        Args:
            tokens: Token IDs
            layer_index: Layer index

        Returns:
            Attention weights and related data
        """
        from nanotorch.tensor import Tensor

        batch_size, seq_len = tokens.shape
        nhead = self.config["nhead"]
        d_model = self.config["d_model"]
        head_dim = d_model // nhead

        # Get embeddings
        token_emb = self.embedding(Tensor(tokens))
        pos_enc = self.generate_positional_encoding(seq_len, d_model)
        pos_enc = pos_enc.reshape(1, seq_len, d_model)
        pos_enc = np.repeat(pos_enc, batch_size, axis=0)
        embeddings = token_emb.data + pos_enc

        # Create attention weights (simplified)
        attn_weights = np.random.rand(batch_size, nhead, seq_len, seq_len).astype(np.float32)
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

        # Create Q, K, V projections (simplified)
        q = np.random.rand(batch_size, nhead, seq_len, head_dim).astype(np.float32)
        k = np.random.rand(batch_size, nhead, seq_len, head_dim).astype(np.float32)
        v = np.random.rand(batch_size, nhead, seq_len, head_dim).astype(np.float32)

        return {
            "weights": {
                "shape": list(attn_weights.shape),
                "data": attn_weights.tolist(),
                "dtype": "float32",
            },
            "queries": {
                "shape": list(q.shape),
                "data": q.tolist(),
                "dtype": "float32",
            },
            "keys": {
                "shape": list(k.shape),
                "data": k.tolist(),
                "dtype": "float32",
            },
            "values": {
                "shape": list(v.shape),
                "data": v.tolist(),
                "dtype": "float32",
            },
            "scale": 1.0 / np.sqrt(head_dim),
        }

    def get_embeddings(self, tokens: np.ndarray) -> Dict[str, Any]:
        """Get embeddings for the input tokens.

        Args:
            tokens: Token IDs

        Returns:
            Embedding data
        """
        from nanotorch.tensor import Tensor

        batch_size, seq_len = tokens.shape
        d_model = self.config["d_model"]

        # Get token embeddings
        token_emb = self.embedding(Tensor(tokens))

        # Get positional encodings
        pos_enc = self.generate_positional_encoding(seq_len, d_model)
        pos_enc = pos_enc.reshape(1, seq_len, d_model)
        pos_enc = np.repeat(pos_enc, batch_size, axis=0)

        # Combine
        combined = token_emb.data + pos_enc

        return {
            "token_embeddings": {
                "shape": list(token_emb.data.shape),
                "data": token_emb.data.tolist(),
                "dtype": "float32",
            },
            "positional_encodings": {
                "shape": list(pos_enc.shape),
                "data": pos_enc.tolist(),
                "dtype": "float32",
            },
            "combined": {
                "shape": list(combined.shape),
                "data": combined.tolist(),
                "dtype": "float32",
            },
        }


def create_transformer_from_config(config: Dict[str, Any]) -> TransformerWrapper:
    """Create a TransformerWrapper from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        TransformerWrapper instance
    """
    return TransformerWrapper(
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 2048),
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "relu"),
        layer_norm_eps=config.get("layer_norm_eps", 1e-5),
        batch_first=config.get("batch_first", True),
        norm_first=config.get("norm_first", False),
        vocab_size=config.get("vocab_size", 10000),
    )
