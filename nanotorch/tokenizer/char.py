"""
Character-level tokenizer implementation.

This module provides a simple character-level tokenizer that splits text into
individual characters.
"""

from typing import List, Dict, Optional
from collections import Counter

from .base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.

    This tokenizer splits text into individual characters and builds a vocabulary
    from the unique characters in the training corpus. It's simple but effective
    for certain tasks and provides a baseline for comparison with other tokenizers.

    Example:
        >>> tokenizer = CharTokenizer(vocab_size=100)
        >>> tokenizer.train(["Hello world"])
        >>> token_ids = tokenizer.encode("Hello")
        >>> print(token_ids)  # [2, 5, 6, 7, 7, 8, 3]  # <sos>, H, e, l, l, o, <eos>
        >>> text = tokenizer.decode(token_ids)
        >>> print(text)  # "Hello"
    """

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize the character tokenizer.

        Args:
            vocab_size: Maximum number of unique characters in the vocabulary
        """
        super().__init__(vocab_size=vocab_size)

    def train(self, texts: List[str]) -> None:
        """
        Train the character tokenizer on a corpus of texts.

        Collects all unique characters from the texts, counts their frequencies,
        and builds a vocabulary with the most frequent characters.

        Args:
            texts: List of text strings to train on
        """
        # Initialize special tokens
        self._initialize_special_tokens()

        # Collect all characters and their frequencies
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)

        # Sort characters by frequency (descending) and add to vocabulary
        # Start from index 4 (after special tokens)
        idx = 4
        for char, freq in char_freq.most_common():
            if idx >= self.vocab_size:
                break
            if char not in self.vocab:  # Avoid duplicates
                self.vocab[char] = idx
                self.token_frequencies[idx] = freq
                self.reverse_vocab[idx] = char
                idx += 1

        self._is_trained = True

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Wraps the token sequence with start-of-sequence (<sos>) and
        end-of-sequence (<eos>) tokens.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs including special tokens
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")

        # Start with SOS token
        tokens = [self.sos_token_id()]

        # Encode each character
        for char in text:
            token_id = self.vocab.get(char, self.unk_token_id())
            tokens.append(token_id)

        # End with EOS token
        tokens.append(self.eos_token_id())

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Skips special tokens (<sos>, <eos>, <pad>) during decoding.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")

        chars = []
        special_ids = {self.pad_token_id(), self.sos_token_id(), self.eos_token_id()}

        for token_id in token_ids:
            if token_id in special_ids:
                continue
            char = self.reverse_vocab.get(token_id, self.UNK_TOKEN)
            chars.append(char)

        return ''.join(chars)

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (characters) without converting to IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of character tokens
        """
        return list(text)

    def get_token_info(self, token_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific token.

        Args:
            token_id: ID of the token to look up

        Returns:
            Dictionary containing token information or None if not found
        """
        if not self._is_trained:
            return None

        if token_id not in self.reverse_vocab:
            return None

        return {
            'id': token_id,
            'text': self.reverse_vocab[token_id],
            'frequency': self.token_frequencies.get(token_id, 0),
            'is_special': token_id in self.special_tokens.values(),
            'type': 'character'
        }

    def encode_with_positions(self, text: str) -> List[Dict]:
        """
        Encode text and return tokens with their positions.

        This is useful for visualization purposes, allowing you to show
        where each token appears in the original text.

        Args:
            text: Input text to encode

        Returns:
            List of dictionaries containing token_id, text, start_position, end_position
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")

        tokens = []
        # Start with SOS token (position -1 to indicate it's not in original text)
        tokens.append({
            'token_id': self.sos_token_id(),
            'text': self.SOS_TOKEN,
            'start_position': -1,
            'end_position': -1,
            'is_special': True
        })

        # Encode each character with its position
        for i, char in enumerate(text):
            token_id = self.vocab.get(char, self.unk_token_id())
            tokens.append({
                'token_id': token_id,
                'text': char if token_id != self.unk_token_id() else self.UNK_TOKEN,
                'start_position': i,
                'end_position': i + 1,
                'is_special': False
            })

        # End with EOS token
        tokens.append({
            'token_id': self.eos_token_id(),
            'text': self.EOS_TOKEN,
            'start_position': -1,
            'end_position': -1,
            'is_special': True
        })

        return tokens
