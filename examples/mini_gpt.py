"""
Mini GPT - A tiny character-level language model using nanotorch.

This example demonstrates training a small GPT-like model on text data.
The model learns to predict the next character in a sequence.

Usage:
    python examples/mini_gpt.py

The model will train on the embedded text and generate sample outputs.
"""

import numpy as np
from nanotorch import Tensor
from nanotorch.nn import (
    Module, Embedding, Linear, LayerNorm, Dropout,
    TransformerEncoderLayer, TransformerEncoder,
    CrossEntropyLoss
)
from nanotorch.optim import AdamW
from nanotorch.optim.lr_scheduler import CosineAnnealingLR
from nanotorch.utils import clip_grad_norm_


class MiniGPT(Module):
    """A minimal GPT-like language model.

    Architecture:
        - Token embeddings
        - Position embeddings
        - Stack of transformer encoder layers (with causal attention)
        - Final layer norm
        - Language model head (linear projection to vocab)

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Dimension of feedforward network.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = Embedding(max_seq_len, d_model)

        self.drop = Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param is not None:
                if "weight" in name:
                    if "embedding" in name:
                        pass
                    elif "lm_head" in name:
                        pass
                    else:
                        pass

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = Tensor(np.arange(T))
        pos_emb = self.pos_embedding(pos)

        x = self.drop(tok_emb + pos_emb)

        x = self.transformer(x, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            if idx.shape[1] > self.max_seq_len:
                idx_cond = idx[:, -self.max_seq_len:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = logits.softmax(dim=-1)
            probs_np = probs.data

            idx_next = []
            for i in range(probs_np.shape[0]):
                idx_next.append(np.random.choice(self.vocab_size, p=probs_np[i]))
            idx_next = Tensor(np.array(idx_next).reshape(-1, 1))

            idx = Tensor(np.concatenate([idx.data, idx_next.data], axis=1))

        return idx


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def train(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list) -> str:
        return "".join(self.idx_to_char[i] for i in indices)


def get_batch(data: np.ndarray, batch_size: int, seq_len: int):
    ix = np.random.randint(0, len(data) - seq_len, batch_size)
    x = np.stack([data[i:i + seq_len] for i in ix])
    y = np.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return Tensor(x), Tensor(y)


def train():
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep: perchance to dream: ay, there's the rub;
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life;
    For who would bear the whips and scorns of time,
    The oppressor's wrong, the proud man's contumely,
    The pangs of despised love, the law's delay,
    The insolence of office and the spurns
    That patient merit of the unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? who would fardels bear,
    To grunt and sweat under a weary life,
    But that the dread of something after death,
    The undiscover'd country from whose bourn
    No traveller returns, puzzles the will
    And makes us rather bear those ills we have
    Than fly to others that we know not of?
    Thus conscience does make cowards of us all;
    And thus the native hue of resolution
    Is sicklied o'er with the pale cast of thought,
    And enterprises of great pith and moment
    With this regard their currents turn awry,
    And lose the name of action.
    """
    text = text * 10

    tokenizer = CharTokenizer()
    tokenizer.train(text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(text)} characters")

    data = np.array(tokenizer.encode(text), dtype=np.int64)

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=128,
        dropout=0.1,
    )

    n_params = sum(p.data.size for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")

    learning_rate = 3e-4
    batch_size = 16
    seq_len = 64
    n_epochs = 100

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=learning_rate * 0.1)

    criterion = CrossEntropyLoss()

    model.train()
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(n_epochs):
        model.train()
        X, Y = get_batch(data, batch_size, seq_len)

        logits = model(X)

        loss = criterion(
            logits.reshape((-1, vocab_size)),
            Y.reshape((-1,))
        )

        optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

    print("-" * 60)
    print("\nGenerating text...\n")

    model.eval()
    start_text = "To be"
    start_ids = tokenizer.encode(start_text)
    idx = Tensor(np.array([start_ids]))

    generated = model.generate(idx, max_new_tokens=100, temperature=0.8)
    generated_text = tokenizer.decode(generated.data[0].tolist())

    print(f"Generated text:\n{generated_text}")
    print("\n" + "=" * 60)
    print("Training complete!")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train()
