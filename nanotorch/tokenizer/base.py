"""
Base tokenizer module for nanotorch.

This module provides the abstract base class for all tokenizer implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    All tokenizer implementations should inherit from this class and implement
    the required methods: train, encode, decode, and get_token_info.

    Attributes:
        vocab_size: Maximum size of the vocabulary
        vocab: Dictionary mapping tokens to IDs
        reverse_vocab: Dictionary mapping IDs to tokens
        token_frequencies: Dictionary mapping token IDs to their frequencies
        special_tokens: Dictionary of special token names to their IDs
    """

    # Special token constants
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Maximum size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.token_frequencies: Dict[int, int] = {}
        self.special_tokens: Dict[str, int] = {}
        self._is_trained = False

    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a corpus of texts.

        This method builds the vocabulary from the given texts.

        Args:
            texts: List of text strings to train on
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens without converting to IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        pass

    def get_token_info(self, token_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific token.

        Args:
            token_id: ID of the token to look up

        Returns:
            Dictionary containing token information or None if not found
        """
        if token_id not in self.reverse_vocab:
            return None

        return {
            'id': token_id,
            'text': self.reverse_vocab[token_id],
            'frequency': self.token_frequencies.get(token_id, 0),
            'is_special': token_id in self.special_tokens.values()
        }

    def get_vocabulary(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the complete vocabulary data.

        Returns:
            Dictionary mapping token IDs to their information
        """
        vocab_size = min(self.vocab_size, len(self.reverse_vocab))
        return {
            token_id: {
                'text': self.reverse_vocab.get(token_id, self.UNK_TOKEN),
                'frequency': self.token_frequencies.get(token_id, 0),
                'is_special': token_id in self.special_tokens.values()
            }
            for token_id in range(vocab_size)
            if token_id in self.reverse_vocab
        }

    def get_vocabulary_size(self) -> int:
        """
        Get the current vocabulary size.

        Returns:
            Number of tokens in the vocabulary
        """
        return len(self.reverse_vocab)

    def is_trained(self) -> bool:
        """
        Check if the tokenizer has been trained.

        Returns:
            True if the tokenizer is trained, False otherwise
        """
        return self._is_trained

    def _initialize_special_tokens(self) -> None:
        """
        Initialize special tokens in the vocabulary.

        This should be called by subclasses during training.
        """
        self.special_tokens = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.vocab = dict(self.special_tokens)
        self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}
        self.token_frequencies = {v: 1 for v in self.special_tokens.values()}

    def get_special_tokens(self) -> Dict[str, int]:
        """
        Get the special token mappings.

        Returns:
            Dictionary mapping special token names to their IDs
        """
        return self.special_tokens.copy()

    def pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self.special_tokens.get(self.PAD_TOKEN, 0)

    def unk_token_id(self) -> int:
        """Get the unknown token ID."""
        return self.special_tokens.get(self.UNK_TOKEN, 1)

    def sos_token_id(self) -> int:
        """Get the start-of-sequence token ID."""
        return self.special_tokens.get(self.SOS_TOKEN, 2)

    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.special_tokens.get(self.EOS_TOKEN, 3)
