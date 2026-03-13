"""Embedding layers for nanotorch."""

import math
from typing import Optional
import numpy as np
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings: Size of the dictionary of embeddings.
        embedding_dim: The size of each embedding vector.
        padding_idx: If specified, the entries at padding_idx do not contribute to
            the gradient and the embedding at padding_idx is not updated during training.
        max_norm: If given, each embedding vector with norm larger than max_norm is
            renormalized to have norm max_norm.
        norm_type: The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq: If given, this will scale gradients by the inverse of
            frequency of the words in the mini-batch. Default False.
        sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.

    Shape:
        - Input: (*), IntTensor or LongTensor of arbitrary shape containing indices
        - Output: (*, embedding_dim), where * is the input shape

    Examples:
        >>> embedding = Embedding(10, 3)
        >>> input = Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> output = embedding(input)
        >>> output.shape
        (2, 4, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
            * math.sqrt(1.0 / embedding_dim),
            requires_grad=True,
        )
        self.register_parameter("weight", self.weight)

        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

    def forward(self, input: Tensor) -> Tensor:
        """Look up embeddings by index.

        Args:
            input: Tensor of indices with arbitrary shape (*).

        Returns:
            Tensor of embeddings with shape (*, embedding_dim).
        """
        from nanotorch.autograd import embedding_lookup as autograd_embedding_lookup

        return autograd_embedding_lookup(
            self.weight,
            input,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
        )

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        return s


class EmbeddingBag(Module):
    """Computes sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.

    For bags of constant length and no per_sample_weights, this is equivalent to Embedding
    followed by torch.sum(dim=0) or torch.mean(dim=0).

    Args:
        num_embeddings: Size of the dictionary of embeddings.
        embedding_dim: The size of each embedding vector.
        mode: "sum", "mean" or "max". Specifies the way to reduce the bag.
            Default: "mean".
        padding_idx: If specified, the entries at padding_idx do not contribute to
            the gradient and the embedding at padding_idx is not updated during training.

    Shape:
        - Input (indices): (N, L) or (N,) where N is batch size and L is sequence length
        - Output: (N, embedding_dim)

    Examples:
        >>> embedding_bag = EmbeddingBag(10, 3, mode='mean')
        >>> input = Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> output = embedding_bag(input)
        >>> output.shape
        (2, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "mean",
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx

        if mode not in ("sum", "mean", "max"):
            raise ValueError(f"mode must be 'sum', 'mean' or 'max', got {mode}")

        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
            * math.sqrt(1.0 / embedding_dim),
            requires_grad=True,
        )
        self.register_parameter("weight", self.weight)

        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

    def forward(self, input: Tensor) -> Tensor:
        """Compute bag embeddings.

        Args:
            input: Tensor of indices with shape (N, L) or (N,).

        Returns:
            Tensor with shape (N, embedding_dim).
        """
        from nanotorch.autograd import embedding_bag as autograd_embedding_bag

        return autograd_embedding_bag(self.weight, input, self.mode, self.padding_idx)

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}, mode='{self.mode}'"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s
