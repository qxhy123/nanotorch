"""
Byte Pair Encoding (BPE) tokenizer implementation.

This module provides a BPE tokenizer that learns subword units by iteratively
merging the most frequent character pairs.
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import json

from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer.

    BPE is a subword tokenization algorithm that starts with characters and
    iteratively merges the most frequent adjacent pairs to create new tokens.
    This allows it to handle rare words by breaking them into subwords while
    keeping common words as single tokens.

    Example:
        >>> tokenizer = BPETokenizer(vocab_size=1000, num_merges=100)
        >>> tokenizer.train(["Hello world", "Hello there"])
        >>> token_ids = tokenizer.encode("hello")
        >>> print(token_ids)  # May be [2, X, Y, 3] or [2, Z, 3] depending on merges
        >>> text = tokenizer.decode(token_ids)
        >>> print(text)  # "hello"
    """

    def __init__(self, vocab_size: int = 10000, num_merges: int = 1000):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab_size: Maximum number of tokens in the vocabulary
            num_merges: Number of merge operations to perform during training
        """
        super().__init__(vocab_size=vocab_size)
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []  # List of merge operations
        self.merge_priority: Dict[Tuple[str, str], int] = {}  # Merge order

    def train(self, texts: List[str]) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.

        Starts with character-level vocabulary and iteratively merges the most
        frequent adjacent character pairs.

        Args:
            texts: List of text strings to train on
        """
        # Initialize special tokens
        self._initialize_special_tokens()

        # Preprocess: convert to list of character sequences
        words = []
        char_freq = Counter()

        for text in texts:
            # Split into words (simple whitespace split for training)
            for word in text.split():
                # Represent as list of characters
                chars = list(word)
                words.append(chars)
                for char in chars:
                    char_freq[char] += 1

        # Build initial vocabulary from characters
        self._build_char_vocabulary(char_freq)

        # Perform BPE merges
        for i in range(self.num_merges):
            # Get the most frequent pair
            pair_freq = self._get_pair_frequency(words)
            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)

            # Stop if we've reached vocab size (excluding special tokens)
            if len(self.vocab) >= self.vocab_size:
                break

            # Record the merge
            self.merges.append(best_pair)
            self.merge_priority[best_pair] = i

            # Merge the pair in all words
            new_symbol = best_pair[0] + best_pair[1]
            words = self._merge_pair(words, best_pair, new_symbol)

            # Add to vocabulary if not already present
            if new_symbol not in self.vocab:
                next_idx = len(self.reverse_vocab)
                if next_idx < self.vocab_size:
                    self.vocab[new_symbol] = next_idx
                    self.reverse_vocab[next_idx] = new_symbol
                    self.token_frequencies[next_idx] = pair_freq[best_pair]

        self._is_trained = True

    def _build_char_vocabulary(self, char_freq: Counter) -> None:
        """
        Build initial character vocabulary from character frequencies.

        Args:
            char_freq: Counter of character frequencies
        """
        idx = 4  # Start after special tokens
        for char, freq in char_freq.most_common():
            if idx >= self.vocab_size:
                break
            if char not in self.vocab:
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char
                self.token_frequencies[idx] = freq
                idx += 1

    def _get_pair_frequency(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent character pairs across all words.

        Args:
            words: List of words represented as character lists

        Returns:
            Dictionary mapping pairs to their frequencies
        """
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        return dict(pairs)

    def _merge_pair(
        self,
        words: List[List[str]],
        pair: Tuple[str, str],
        new_symbol: str
    ) -> List[List[str]]:
        """
        Merge all occurrences of a pair in all words.

        Args:
            words: List of words represented as symbol lists
            pair: The pair to merge
            new_symbol: The new symbol to replace the pair with

        Returns:
            Updated list of words
        """
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs using BPE.

        Applies the learned merge operations to encode text efficiently.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs including special tokens
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")

        # Start with SOS token
        tokens = [self.sos_token_id()]

        # Encode each word
        for word in text.split():
            word_tokens = self._encode_word(word.lower())
            tokens.extend(word_tokens)

        # End with EOS token
        tokens.append(self.eos_token_id())

        return tokens

    def _encode_word(self, word: str) -> List[int]:
        """
        Encode a single word using BPE.

        Starts with characters and applies merge operations to create
        longer subword tokens.

        Args:
            word: Single word to encode

        Returns:
            List of token IDs for the word
        """
        if len(word) == 0:
            return [self.unk_token_id()]

        # Start with characters
        word_tokens = list(word)

        # Apply merges in priority order
        # We need to repeatedly scan and apply merges
        while len(word_tokens) > 1:
            merged = False
            for pair in self.merges:
                # Find and merge this pair
                for i in range(len(word_tokens) - 1):
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        # Check if the merged symbol is in vocab
                        new_symbol = pair[0] + pair[1]
                        if new_symbol in self.vocab:
                            word_tokens[i:i+2] = [new_symbol]
                            merged = True
                            break
                if merged:
                    break

            if not merged:
                break

        # Convert to token IDs
        token_ids = []
        for token in word_tokens:
            token_id = self.vocab.get(token, self.unk_token_id())
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Concatenates all tokens without spaces (BPE typically doesn't use
        spaces between subwords of the same word).

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")

        tokens = []
        special_ids = {self.pad_token_id(), self.sos_token_id(), self.eos_token_id()}

        # Decode tokens, handling word boundaries
        current_word = []

        for token_id in token_ids:
            if token_id in special_ids:
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                continue

            token = self.reverse_vocab.get(token_id, self.UNK_TOKEN)

            # Check if this might be a word boundary
            # In BPE, we can detect boundaries by checking if the token
            # could start a word (starts with a character that wasn't created by merge)
            if self._is_word_start(token):
                if current_word:
                    tokens.append(''.join(current_word))
                current_word = [token]
            else:
                current_word.append(token)

        # Don't forget the last word
        if current_word:
            tokens.append(''.join(current_word))

        return ' '.join(tokens)

    def _is_word_start(self, token: str) -> bool:
        """
        Check if a token likely represents the start of a word.

        This is a heuristic: single characters are always word starts,
        and we check if the token begins with a character that's not
        commonly part of a merge.

        Args:
            token: Token string to check

        Returns:
            True if this token likely starts a word
        """
        if len(token) == 1:
            return True

        # Check if token starts with a whitespace (would indicate word boundary)
        return token[0].isalpha() and token[0].islower()

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into subword tokens without converting to IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of subword tokens
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before tokenizing")

        tokens = []
        for word in text.split():
            word_tokens = self._tokenize_word(word.lower())
            tokens.extend(word_tokens)
        return tokens

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.

        Args:
            word: Single word to tokenize

        Returns:
            List of subword tokens
        """
        if len(word) == 0:
            return [self.UNK_TOKEN]

        # Start with characters
        word_tokens = list(word)

        # Apply merges
        while len(word_tokens) > 1:
            merged = False
            for pair in self.merges:
                for i in range(len(word_tokens) - 1):
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        new_symbol = pair[0] + pair[1]
                        if new_symbol in self.vocab:
                            word_tokens[i:i+2] = [new_symbol]
                            merged = True
                            break
                if merged:
                    break
            if not merged:
                break

        return word_tokens

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

        token = self.reverse_vocab[token_id]
        return {
            'id': token_id,
            'text': token,
            'frequency': self.token_frequencies.get(token_id, 0),
            'is_special': token_id in self.special_tokens.values(),
            'type': 'subword',
            'length': len(token)
        }

    def encode_with_positions(self, text: str) -> List[Dict]:
        """
        Encode text and return tokens with their positions.

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

        # Process each word
        word_start = 0
        for word in text.split():
            # Find this word in the original text
            word_pos = text.lower().find(word.lower(), word_start)
            if word_pos == -1:
                word_pos = word_start

            word_end = word_pos + len(word)

            # Encode the word
            word_token_ids = self._encode_word(word.lower())

            # For subwords, distribute positions across the word
            for i, token_id in enumerate(word_token_ids):
                token_text = self.reverse_vocab.get(token_id, self.UNK_TOKEN)
                # Approximate positions for subwords
                sub_start = word_pos + (i * len(word) // len(word_token_ids))
                sub_end = word_pos + ((i + 1) * len(word) // len(word_token_ids))

                tokens.append({
                    'token_id': token_id,
                    'text': token_text,
                    'start_position': sub_start,
                    'end_position': sub_end,
                    'is_special': False
                })

            word_start = word_end

        # End with EOS token
        tokens.append({
            'token_id': self.eos_token_id(),
            'text': self.EOS_TOKEN,
            'start_position': -1,
            'end_position': -1,
            'is_special': True
        })

        return tokens

    def get_merges(self) -> List[Tuple[str, str]]:
        """
        Get the list of merge operations learned during training.

        Returns:
            List of merged pairs in order of priority
        """
        return self.merges.copy()

    def save_merges(self, filepath: str) -> None:
        """
        Save merge operations to a file.

        Args:
            filepath: Path to save the merges
        """
        with open(filepath, 'w') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

    def load_merges(self, filepath: str) -> None:
        """
        Load merge operations from a file.

        Args:
            filepath: Path to load the merges from
        """
        self.merges = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))

        # Rebuild merge priority
        self.merge_priority = {pair: i for i, pair in enumerate(self.merges)}
