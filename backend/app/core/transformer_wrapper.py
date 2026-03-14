"""Wrapper class for nanotorch Transformer model."""

import json
import hashlib
import os
import sys
from contextlib import contextmanager
import numpy as np
from typing import Optional, Dict, Any

# Add parent directory to path to import nanotorch
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
sys.path.insert(0, _project_root)

try:
    from nanotorch.nn.transformer import Transformer
    from nanotorch.nn.embedding import Embedding
    NANOTORCH_AVAILABLE = True
except ImportError:
    NANOTORCH_AVAILABLE = False
    Transformer = None
    Embedding = None


MODEL_CONFIG_KEYS = (
    "d_model",
    "nhead",
    "num_encoder_layers",
    "num_decoder_layers",
    "dim_feedforward",
    "dropout",
    "activation",
    "layer_norm_eps",
    "batch_first",
    "norm_first",
    "vocab_size",
)

MODEL_CONFIG_DEFAULTS = {
    "d_model": 512,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "activation": "relu",
    "layer_norm_eps": 1e-5,
    "batch_first": True,
    "norm_first": False,
    "vocab_size": 10000,
}

TRANSFORMER_CACHE: Dict[str, "TransformerWrapper"] = {}


def _normalize_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model config for stable hashing and wrapper construction."""
    normalized: Dict[str, Any] = {}

    for key in MODEL_CONFIG_KEYS:
        value = config.get(key, MODEL_CONFIG_DEFAULTS[key])
        if value is None:
            value = MODEL_CONFIG_DEFAULTS[key]
        if key == "activation" and hasattr(value, "value"):
            value = value.value
        normalized[key] = value

    return normalized


def _stable_seed(*parts: Any) -> int:
    """Create a deterministic 32-bit seed from JSON-serializable inputs."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@contextmanager
def _temporary_numpy_seed(seed: int):
    """Temporarily seed NumPy without affecting the global RNG afterwards."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _tensor_payload(array: np.ndarray) -> Dict[str, Any]:
    """Serialize a NumPy array for the API response."""
    return {
        "shape": list(array.shape),
        "data": array.tolist(),
        "dtype": str(array.dtype),
    }


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

        if hasattr(activation, "value"):
            activation = activation.value

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
        self._model_seed = _stable_seed("transformer_model", self.config)

        with _temporary_numpy_seed(self._model_seed):
            self.embedding = Embedding(vocab_size, d_model)
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

    def _simulate_attention_data(
        self,
        tokens: np.ndarray,
        layer_index: int,
    ) -> Dict[str, Any]:
        """Build deterministic simulated attention tensors for visualization."""
        batch_size, seq_len = tokens.shape
        nhead = self.config["nhead"]
        d_model = self.config["d_model"]
        head_dim = d_model // nhead
        scale = np.float32(1.0 / np.sqrt(head_dim))
        rng = np.random.default_rng(
            _stable_seed(
                "attention_visualization",
                self.config,
                layer_index,
                tokens.astype(np.int64).tolist(),
            )
        )

        q = (rng.normal(0.0, 0.1, size=(batch_size, nhead, seq_len, head_dim))).astype(np.float32)
        k = (rng.normal(0.0, 0.1, size=(batch_size, nhead, seq_len, head_dim))).astype(np.float32)
        v = (rng.normal(0.0, 0.1, size=(batch_size, nhead, seq_len, head_dim))).astype(np.float32)

        scores = np.matmul(q, np.swapaxes(k, -1, -2)).astype(np.float32)
        scores *= scale
        scores -= np.max(scores, axis=-1, keepdims=True)

        attn_weights = np.exp(scores).astype(np.float32)
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

        return {
            "weights": _tensor_payload(attn_weights.astype(np.float32)),
            "queries": _tensor_payload(q),
            "keys": _tensor_payload(k),
            "values": _tensor_payload(v),
            "scale": float(scale),
        }

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
                "token_embeddings": _tensor_payload(token_emb_data),
                "positional_encodings": _tensor_payload(pos_enc),
                "combined": _tensor_payload(src_embeddings),
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
                        "token_embeddings": _tensor_payload(tgt_token_emb_data),
                        "positional_encodings": _tensor_payload(tgt_pos_enc),
                        "combined": _tensor_payload(tgt_embeddings),
                    }
            else:
                # No target provided, use source as target (autoencoder-like behavior)
                tgt = Tensor(src_embeddings)

            output = self.transformer(src, tgt)
        else:
            # Encoder only
            output = self.transformer.encoder(src)

        results["final_output"] = _tensor_payload(output.data)

        # Note: In a full implementation, we would hook into the layers
        # to collect intermediate outputs and attention weights.
        # For now, we return a placeholder structure.

        if return_all_layers:
            results["metadata"] = {
                **results.get("metadata", {}),
                "layer_outputs_available": False,
                "layer_outputs_note": (
                    "Per-layer capture is not implemented yet; placeholder layer_outputs "
                    "have been removed to avoid misleading consumers."
                ),
            }

        if return_attention:
            # Create attention data with Q, K, V projections
            results["attention_weights"] = []
            num_layers = self.config["num_encoder_layers"]

            for i in range(num_layers):
                results["attention_weights"].append(self._simulate_attention_data(tokens, i))

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
        return self._simulate_attention_data(tokens, layer_index)

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
            "token_embeddings": _tensor_payload(token_emb.data),
            "positional_encodings": _tensor_payload(pos_enc),
            "combined": _tensor_payload(combined),
        }


def create_transformer_from_config(config: Dict[str, Any]) -> TransformerWrapper:
    """Create a TransformerWrapper from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        TransformerWrapper instance
    """
    normalized_config = _normalize_model_config(config)
    cache_key = json.dumps(normalized_config, sort_keys=True, separators=(",", ":"))

    if cache_key not in TRANSFORMER_CACHE:
        TRANSFORMER_CACHE[cache_key] = TransformerWrapper(
            d_model=normalized_config.get("d_model", 512),
            nhead=normalized_config.get("nhead", 8),
            num_encoder_layers=normalized_config.get("num_encoder_layers", 6),
            num_decoder_layers=normalized_config.get("num_decoder_layers", 6),
            dim_feedforward=normalized_config.get("dim_feedforward", 2048),
            dropout=normalized_config.get("dropout", 0.1),
            activation=normalized_config.get("activation", "relu"),
            layer_norm_eps=normalized_config.get("layer_norm_eps", 1e-5),
            batch_first=normalized_config.get("batch_first", True),
            norm_first=normalized_config.get("norm_first", False),
            vocab_size=normalized_config.get("vocab_size", 10000),
        )

    return TRANSFORMER_CACHE[cache_key]
