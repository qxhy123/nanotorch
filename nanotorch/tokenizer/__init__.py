"""
Tokenizer module for nanotorch.

This module provides various tokenization algorithms for natural language processing,
including character-level, word-level, and subword (BPE) tokenizers.
"""

from .base import BaseTokenizer
from .char import CharTokenizer
from .word import WordTokenizer
from .bpe import BPETokenizer

__all__ = [
    'BaseTokenizer',
    'CharTokenizer',
    'WordTokenizer',
    'BPETokenizer',
]


# Convenience function to get a tokenizer by type
def get_tokenizer(tokenizer_type: str, **kwargs):
    """
    Get a tokenizer instance by type.

    Args:
        tokenizer_type: Type of tokenizer ('char', 'word', or 'bpe')
        **kwargs: Additional arguments to pass to the tokenizer constructor

    Returns:
        Tokenizer instance

    Raises:
        ValueError: If tokenizer_type is not recognized
    """
    tokenizers = {
        'char': CharTokenizer,
        'word': WordTokenizer,
        'bpe': BPETokenizer,
    }

    tokenizer_class = tokenizers.get(tokenizer_type.lower())
    if tokenizer_class is None:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Available types: {list(tokenizers.keys())}"
        )

    return tokenizer_class(**kwargs)
