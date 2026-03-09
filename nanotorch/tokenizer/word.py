"""
Word-level tokenizer implementation.

This module provides a word-level tokenizer that splits text into words using
regular expressions.
"""

import re
from typing import List, Dict, Optional
from collections import Counter

from .base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    Word-level tokenizer.

    This tokenizer splits text into words using a regular expression pattern.
    It handles punctuation as separate tokens and preserves common punctuation marks.

    The word pattern matches:
    - Sequences of lowercase letters (words)
    - Individual punctuation marks (!?.,'"-)
    - Sequences of digits (numbers)

    Example:
        >>> tokenizer = WordTokenizer(vocab_size=1000)
        >>> tokenizer.train(["Hello, world! This is a test."])
        >>> token_ids = tokenizer.encode("Hello world")
        >>> print(token_ids)  # [2, X, Y, 3]  # <sos>, hello, world, <eos>
        >>> text = tokenizer.decode(token_ids)
        >>> print(text)  # "hello world"
    """

    # Pattern to match words and punctuation
    WORD_PATTERN = re.compile(r"[a-z]+|[!?.,'\"-]|[0-9]+")

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize the word tokenizer.

        Args:
            vocab_size: Maximum number of unique words in the vocabulary
        """
        super().__init__(vocab_size=vocab_size)

    def train(self, texts: List[str]) -> None:
        """
        Train the word tokenizer on a corpus of texts.

        Splits texts into words, counts their frequencies, and builds a vocabulary
        with the most frequent words.

        Args:
            texts: List of text strings to train on
        """
        # Initialize special tokens
        self._initialize_special_tokens()

        # Collect all words and their frequencies
        word_freq = Counter()
        for text in texts:
            # Convert to lowercase and extract words
            words = self.WORD_PATTERN.findall(text.lower())
            word_freq.update(words)

        # Sort words by frequency (descending) and add to vocabulary
        # Start from index 4 (after special tokens)
        idx = 4
        for word, freq in word_freq.most_common():
            if idx >= self.vocab_size:
                break
            if word not in self.vocab:  # Avoid duplicates
                self.vocab[word] = idx
                self.token_frequencies[idx] = freq
                self.reverse_vocab[idx] = word
                idx += 1

        self._is_trained = True

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Wraps the token sequence with start-of-sequence (<sos>) and
        end-of-sequence (<eos>) tokens. Converts text to lowercase.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs including special tokens
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")

        # Extract words (lowercase)
        words = self.WORD_PATTERN.findall(text.lower())

        # Start with SOS token
        tokens = [self.sos_token_id()]

        # Encode each word
        for word in words:
            token_id = self.vocab.get(word, self.unk_token_id())
            tokens.append(token_id)

        # End with EOS token
        tokens.append(self.eos_token_id())

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Skips special tokens (<sos>, <eos>, <pad>) during decoding.
        Joins words with spaces.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")

        words = []
        special_ids = {self.pad_token_id(), self.sos_token_id(), self.eos_token_id()}

        for token_id in token_ids:
            if token_id in special_ids:
                continue
            word = self.reverse_vocab.get(token_id, self.UNK_TOKEN)
            words.append(word)

        return ' '.join(words)

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (words) without converting to IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of word tokens
        """
        return self.WORD_PATTERN.findall(text.lower())

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
            'type': 'word'
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
        # Start with SOS token
        tokens.append({
            'token_id': self.sos_token_id(),
            'text': self.SOS_TOKEN,
            'start_position': -1,
            'end_position': -1,
            'is_special': True
        })

        # Find all matches with their positions
        lower_text = text.lower()
        for match in self.WORD_PATTERN.finditer(lower_text):
            word = match.group()
            token_id = self.vocab.get(word, self.unk_token_id())
            tokens.append({
                'token_id': token_id,
                'text': word if token_id != self.unk_token_id() else self.UNK_TOKEN,
                'start_position': match.start(),
                'end_position': match.end(),
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

    def get_vocab_statistics(self) -> Dict:
        """
        Get statistics about the vocabulary.

        Returns:
            Dictionary containing vocabulary statistics
        """
        if not self._is_trained:
            return {}

        total_freq = sum(
            freq for token_id, freq in self.token_frequencies.items()
            if token_id not in self.special_tokens.values()
        )

        return {
            'vocab_size': len(self.reverse_vocab),
            'total_tokens': total_freq,
            'num_special_tokens': len(self.special_tokens),
            'most_common': [
                {'text': self.reverse_vocab[token_id], 'frequency': freq}
                for token_id, freq in sorted(
                    self.token_frequencies.items(),
                    key=lambda x: -x[1]
                )[:10]
                if token_id not in self.special_tokens.values()
            ]
        }
