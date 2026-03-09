"""API routes for tokenizer operations."""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from nanotorch.tokenizer import (
    BaseTokenizer,
    CharTokenizer,
    WordTokenizer,
    BPETokenizer,
    get_tokenizer,
)

router = APIRouter(prefix="/api/v1/tokenizer", tags=["tokenizer"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TokenizerType(str, Enum):
    """Supported tokenizer types."""
    char = "char"
    word = "word"
    bpe = "bpe"


class TokenizeRequest(BaseModel):
    """Request to tokenize text."""
    text: str = Field(..., description="Text to tokenize", min_length=1)
    tokenizer_type: TokenizerType = Field(
        default=TokenizerType.char,
        description="Type of tokenizer to use"
    )
    vocab_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum vocabulary size"
    )
    num_merges: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Number of BPE merge operations (only for BPE tokenizer)"
    )
    training_texts: Optional[List[str]] = Field(
        default=None,
        description="Optional training texts. If not provided, uses the input text for training."
    )


class TokenPosition(BaseModel):
    """Position information for a token."""
    start_position: int = Field(..., description="Start position in original text")
    end_position: int = Field(..., description="End position in original text")


class TokenInfo(BaseModel):
    """Information about a single token."""
    token_id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token text")
    frequency: int = Field(..., description="Token frequency in training corpus")
    is_special: bool = Field(..., description="Whether this is a special token")
    start_position: Optional[int] = Field(None, description="Start position in original text")
    end_position: Optional[int] = Field(None, description="End position in original text")


class VocabularyTokenInfo(BaseModel):
    """Information about a token in the vocabulary."""
    id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token text")
    frequency: int = Field(..., description="Token frequency")
    is_special: bool = Field(..., description="Whether this is a special token")


class TokenizeResponse(BaseModel):
    """Response from tokenization."""
    success: bool = Field(..., description="Whether tokenization was successful")
    token_ids: List[int] = Field(..., description="List of token IDs")
    tokens: List[str] = Field(..., description="List of token strings")
    token_details: List[TokenInfo] = Field(..., description="Detailed token information")
    vocabulary_summary: Dict[str, Any] = Field(..., description="Vocabulary summary")
    tokenizer_type: str = Field(..., description="Type of tokenizer used")
    error: Optional[str] = Field(None, description="Error message if failed")


class VocabularyRequest(BaseModel):
    """Request to get vocabulary."""
    tokenizer_type: TokenizerType = Field(
        default=TokenizerType.char,
        description="Type of tokenizer"
    )
    vocab_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum vocabulary size"
    )
    training_texts: Optional[List[str]] = Field(
        default=None,
        description="Training texts. Uses default corpus if not provided."
    )


class VocabularyResponse(BaseModel):
    """Response with vocabulary data."""
    success: bool = Field(..., description="Whether request was successful")
    vocab_size: int = Field(..., description="Total vocabulary size")
    tokens: List[VocabularyTokenInfo] = Field(..., description="All tokens in vocabulary")
    special_tokens: Dict[str, int] = Field(..., description="Special token IDs")
    tokenizer_type: str = Field(..., description="Type of tokenizer")
    error: Optional[str] = Field(None, description="Error message if failed")


class TokenDetailRequest(BaseModel):
    """Request to get detailed token information."""
    token_id: int = Field(..., ge=0, description="Token ID to look up")
    tokenizer_type: TokenizerType = Field(
        default=TokenizerType.char,
        description="Type of tokenizer"
    )
    vocab_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum vocabulary size"
    )
    training_texts: Optional[List[str]] = Field(
        default=None,
        description="Training texts. Uses default corpus if not provided."
    )


class TokenDetailResponse(BaseModel):
    """Response with detailed token information."""
    success: bool = Field(..., description="Whether request was successful")
    token_info: Optional[Dict[str, Any]] = Field(None, description="Token information")
    tokenizer_type: str = Field(..., description="Type of tokenizer")
    error: Optional[str] = Field(None, description="Error message if failed")


class DecodeRequest(BaseModel):
    """Request to decode token IDs."""
    token_ids: List[int] = Field(..., description="Token IDs to decode")
    tokenizer_type: TokenizerType = Field(
        default=TokenizerType.char,
        description="Type of tokenizer"
    )
    vocab_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum vocabulary size"
    )
    training_texts: Optional[List[str]] = Field(
        default=None,
        description="Training texts. Uses default corpus if not provided."
    )


class DecodeResponse(BaseModel):
    """Response from decoding."""
    success: bool = Field(..., description="Whether decoding was successful")
    text: str = Field(..., description="Decoded text")
    tokenizer_type: str = Field(..., description="Type of tokenizer used")
    error: Optional[str] = Field(None, description="Error message if failed")


class CompareRequest(BaseModel):
    """Request to compare different tokenizers."""
    text: str = Field(..., description="Text to tokenize", min_length=1)
    vocab_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum vocabulary size for all tokenizers"
    )
    num_merges: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Number of BPE merge operations"
    )
    training_texts: Optional[List[str]] = Field(
        default=None,
        description="Optional training texts"
    )


class ComparisonResult(BaseModel):
    """Result from comparing tokenizers."""
    tokenizer_type: str = Field(..., description="Tokenizer type")
    num_tokens: int = Field(..., description="Number of tokens produced")
    tokens: List[str] = Field(..., description="Token strings")
    token_ids: List[int] = Field(..., description="Token IDs")
    oov_count: int = Field(..., description="Number of out-of-vocabulary tokens")


class CompareResponse(BaseModel):
    """Response from tokenizer comparison."""
    success: bool = Field(..., description="Whether comparison was successful")
    comparisons: List[ComparisonResult] = Field(..., description="Comparison results for each tokenizer")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Default Training Corpus
# ============================================================================

DEFAULT_TRAINING_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing deals with the interaction between computers and human language.",
    "Transformers are neural network architectures that use attention mechanisms.",
    "Tokenization is the process of breaking text into smaller units called tokens.",
    "Deep learning models require large amounts of training data.",
    "Attention mechanisms allow models to focus on different parts of the input.",
    "Embeddings represent words as dense vectors in a continuous space.",
    "The Transformer architecture revolutionized natural language processing.",
    "Byte Pair Encoding is a subword tokenization algorithm.",
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_train_tokenizer(
    tokenizer_type: str,
    training_texts: Optional[List[str]],
    vocab_size: int,
    num_merges: int = 1000
) -> BaseTokenizer:
    """
    Get an existing trained tokenizer or train a new one.

    Args:
        tokenizer_type: Type of tokenizer
        training_texts: Texts to train on (uses default if None)
        vocab_size: Maximum vocabulary size
        num_merges: Number of BPE merges

    Returns:
        Trained tokenizer instance
    """
    # Get tokenizer
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size, num_merges=num_merges)
    elif tokenizer_type == "word":
        tokenizer = WordTokenizer(vocab_size=vocab_size)
    else:
        tokenizer = CharTokenizer(vocab_size=vocab_size)

    # Train the tokenizer
    texts = training_texts if training_texts else DEFAULT_TRAINING_CORPUS
    tokenizer.train(texts)

    return tokenizer


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """Tokenize text using the specified tokenizer.

    Args:
        request: Tokenization request with text, tokenizer type, and options

    Returns:
        Tokenization response with token IDs, token strings, and metadata
    """
    try:
        # Create and train tokenizer
        tokenizer = get_or_train_tokenizer(
            tokenizer_type=request.tokenizer_type.value,
            training_texts=request.training_texts,
            vocab_size=request.vocab_size,
            num_merges=request.num_merges
        )

        # Encode text
        token_ids = tokenizer.encode(request.text)
        tokens = [tokenizer.reverse_vocab.get(tid, '<unk>') for tid in token_ids]

        # Get detailed token information with positions
        token_details_data = tokenizer.encode_with_positions(request.text)

        token_details = [
            TokenInfo(
                token_id=t['token_id'],
                text=t['text'],
                frequency=tokenizer.token_frequencies.get(t['token_id'], 0),
                is_special=t['is_special'],
                start_position=t.get('start_position'),
                end_position=t.get('end_position')
            )
            for t in token_details_data
        ]

        # Get vocabulary summary
        vocabulary = tokenizer.get_vocabulary()
        vocabulary_summary = {
            "vocab_size": len(tokenizer.reverse_vocab),
            "special_tokens": tokenizer.get_special_tokens(),
            "most_common": sorted(
                [
                    {"id": tid, "text": info["text"], "frequency": info["frequency"]}
                    for tid, info in vocabulary.items()
                    if not info["is_special"]
                ],
                key=lambda x: -x["frequency"]
            )[:10]
        }

        return TokenizeResponse(
            success=True,
            token_ids=token_ids,
            tokens=tokens,
            token_details=token_details,
            vocabulary_summary=vocabulary_summary,
            tokenizer_type=request.tokenizer_type.value
        )

    except Exception as e:
        return TokenizeResponse(
            success=False,
            token_ids=[],
            tokens=[],
            token_details=[],
            vocabulary_summary={},
            tokenizer_type=request.tokenizer_type.value,
            error=str(e)
        )


@router.post("/vocabulary", response_model=VocabularyResponse)
async def get_vocabulary(request: VocabularyRequest):
    """Get the vocabulary for a tokenizer.

    Args:
        request: Vocabulary request with tokenizer type and options

    Returns:
        Vocabulary response with all tokens and their information
    """
    try:
        # Create and train tokenizer
        tokenizer = get_or_train_tokenizer(
            tokenizer_type=request.tokenizer_type.value,
            training_texts=request.training_texts,
            vocab_size=request.vocab_size
        )

        # Get vocabulary
        vocabulary = tokenizer.get_vocabulary()

        # Convert to response format
        tokens = [
            VocabularyTokenInfo(
                id=token_id,
                text=info["text"],
                frequency=info["frequency"],
                is_special=info["is_special"]
            )
            for token_id, info in vocabulary.items()
        ]

        return VocabularyResponse(
            success=True,
            vocab_size=len(tokenizer.reverse_vocab),
            tokens=tokens,
            special_tokens=tokenizer.get_special_tokens(),
            tokenizer_type=request.tokenizer_type.value
        )

    except Exception as e:
        return VocabularyResponse(
            success=False,
            vocab_size=0,
            tokens=[],
            special_tokens={},
            tokenizer_type=request.tokenizer_type.value,
            error=str(e)
        )


@router.post("/token-detail", response_model=TokenDetailResponse)
async def get_token_detail(request: TokenDetailRequest):
    """Get detailed information about a specific token.

    Args:
        request: Token detail request

    Returns:
        Token detail response
    """
    try:
        # Create and train tokenizer
        tokenizer = get_or_train_tokenizer(
            tokenizer_type=request.tokenizer_type.value,
            training_texts=request.training_texts,
            vocab_size=request.vocab_size
        )

        # Get token info
        token_info = tokenizer.get_token_info(request.token_id)

        if token_info is None:
            return TokenDetailResponse(
                success=False,
                token_info=None,
                tokenizer_type=request.tokenizer_type.value,
                error=f"Token ID {request.token_id} not found in vocabulary"
            )

        return TokenDetailResponse(
            success=True,
            token_info=token_info,
            tokenizer_type=request.tokenizer_type.value
        )

    except Exception as e:
        return TokenDetailResponse(
            success=False,
            token_info=None,
            tokenizer_type=request.tokenizer_type.value,
            error=str(e)
        )


@router.post("/decode", response_model=DecodeResponse)
async def decode(request: DecodeRequest):
    """Decode token IDs back into text.

    Args:
        request: Decode request with token IDs and tokenizer type

    Returns:
        Decoded text
    """
    try:
        # Create and train tokenizer
        tokenizer = get_or_train_tokenizer(
            tokenizer_type=request.tokenizer_type.value,
            training_texts=request.training_texts,
            vocab_size=request.vocab_size
        )

        # Decode
        text = tokenizer.decode(request.token_ids)

        return DecodeResponse(
            success=True,
            text=text,
            tokenizer_type=request.tokenizer_type.value
        )

    except Exception as e:
        return DecodeResponse(
            success=False,
            text="",
            tokenizer_type=request.tokenizer_type.value,
            error=str(e)
        )


@router.post("/compare", response_model=CompareResponse)
async def compare_tokenizers(request: CompareRequest):
    """Compare different tokenizers on the same text.

    Args:
        request: Comparison request with text and options

    Returns:
        Comparison results for each tokenizer type
    """
    try:
        comparisons = []

        for tokenizer_type in ["char", "word", "bpe"]:
            # Create and train tokenizer
            tokenizer = get_or_train_tokenizer(
                tokenizer_type=tokenizer_type,
                training_texts=request.training_texts,
                vocab_size=request.vocab_size,
                num_merges=request.num_merges
            )

            # Encode
            token_ids = tokenizer.encode(request.text)
            tokens = [tokenizer.reverse_vocab.get(tid, '<unk>') for tid in token_ids]

            # Count OOV (unknown) tokens
            oov_count = sum(1 for tid in token_ids if tid == tokenizer.unk_token_id())

            comparisons.append(
                ComparisonResult(
                    tokenizer_type=tokenizer_type,
                    num_tokens=len(token_ids),
                    tokens=tokens,
                    token_ids=token_ids,
                    oov_count=oov_count
                )
            )

        return CompareResponse(
            success=True,
            comparisons=comparisons
        )

    except Exception as e:
        return CompareResponse(
            success=False,
            comparisons=[],
            error=str(e)
        )


@router.get("/types")
async def get_tokenizer_types():
    """Get available tokenizer types.

    Returns:
        List of available tokenizer types with descriptions
    """
    return {
        "success": True,
        "tokenizer_types": [
            {
                "type": "char",
                "name": "Character-level",
                "description": "Splits text into individual characters. Simple but results in long sequences."
            },
            {
                "type": "word",
                "name": "Word-level",
                "description": "Splits text into words using regex. Handles punctuation separately."
            },
            {
                "type": "bpe",
                "name": "Byte Pair Encoding",
                "description": "Subword tokenization that learns merge operations. Balances vocabulary size and sequence length."
            }
        ]
    }
