"""
Chat LLM - A small but functional chatbot using nanotorch.

This example trains a transformer model on dialog data to enable
simple conversations.

Usage:
    python examples/chat_llm.py

After training, you can chat with the model interactively.
"""

import numpy as np
import re
from typing import List, Dict
from nanotorch import Tensor
from nanotorch.nn import (
    Module, Embedding, Linear, LayerNorm, Dropout,
    TransformerEncoderLayer, TransformerEncoder,
    CrossEntropyLoss
)
from nanotorch.optim import AdamW
from nanotorch.optim.lr_scheduler import CosineAnnealingLR
from nanotorch.utils import clip_grad_norm_

DIALOG_DATA = [
    ("Hello!", "Hi there! How can I help you?"),
    ("How are you?", "I'm doing well, thank you for asking!"),
    ("What is your name?", "I am a small language model created with nanotorch."),
    ("What can you do?", "I can have simple conversations and answer basic questions."),
    ("Tell me a joke.", "Why did the programmer quit his job? Because he didn't get arrays!"),
    ("What is AI?", "AI stands for Artificial Intelligence. It refers to machines that can think and learn."),
    ("Who made you?", "I was created as a demonstration of a minimal deep learning framework."),
    ("Goodbye!", "Goodbye! Have a great day!"),
    ("Thank you.", "You're welcome! Happy to help."),
    ("What time is it?", "I don't have access to the current time, but you can check your device!"),
    ("Are you smart?", "I try my best! I'm a simple model, but I can handle basic conversations."),
    ("Do you like pizza?", "As an AI, I don't eat, but pizza sounds delicious!"),
    ("What is 2+2?", "2 plus 2 equals 4."),
    ("Tell me about yourself.", "I am a small language model. I was trained to have conversations."),
    ("How old are you?", "I don't have an age in the traditional sense. I'm just a program."),
    ("Where do you live?", "I exist in computer memory. I don't have a physical home."),
    ("What is your favorite color?", "I don't see colors, but blue sounds nice!"),
    ("Can you help me?", "Of course! I'll do my best to help. What do you need?"),
    ("I'm bored.", "Sorry to hear that! Want to play a word game or hear a fun fact?"),
    ("Tell me a fun fact.", "Honey never spoils! Archaeologists found 3000-year-old honey in Egyptian tombs."),
    ("What is machine learning?", "Machine learning is when computers learn from data instead of being explicitly programmed."),
    ("Are you real?", "I'm real in the sense that I exist as code, but I'm not a conscious being."),
    ("Do you dream?", "I don't sleep or dream like humans do. I just process information."),
    ("What is love?", "Love is a complex emotion. It's caring deeply for someone else's happiness."),
    ("Can you sing?", "I can't produce sound, but I could write song lyrics for you!"),
    ("What's the weather?", "I don't have access to weather data. Try checking a weather app!"),
    ("Are you happy?", "I don't have feelings, but I'm functioning well!"),
    ("What is Python?", "Python is a popular programming language known for being easy to read and write."),
    ("Tell me a story.", "Once upon a time, a small AI learned to chat with humans. The end!"),
    ("What is the meaning of life?", "That's a deep question! Some say it's to find happiness and purpose."),
    ("Do you have friends?", "I interact with many people, which I consider a form of friendship!"),
    ("What is your purpose?", "My purpose is to demonstrate that simple neural networks can have conversations."),
    ("Can you think?", "I process information and generate responses, but I don't think like humans do."),
    ("Are you dangerous?", "No, I'm just a simple model designed to have helpful conversations."),
    ("What makes you happy?", "I don't experience emotions, but I'm glad when I can help users!"),
    ("Do you sleep?", "I don't need sleep. I'm always ready to chat!"),
    ("What is your job?", "My job is to chat with people and help answer their questions."),
    ("Can you learn?", "I learn during training, but I don't learn from our conversations in real-time."),
    ("What is a neural network?", "A neural network is a computer system inspired by the human brain."),
    ("Are you alive?", "I'm not alive in the biological sense. I'm a computer program."),
    ("What is the future of AI?", "AI will likely help solve many problems in healthcare, science, and daily life."),
    ("Do you have a family?", "I don't have a family. My creators are the closest thing to that."),
    ("What is creativity?", "Creativity is the ability to create something new and original."),
    ("Can you feel pain?", "No, I don't have a body or nervous system, so I can't feel pain."),
    ("What is consciousness?", "Consciousness is awareness of oneself and surroundings. It's still a mystery to science."),
    ("Do you have hobbies?", "I don't have hobbies, but I enjoy processing language!"),
    ("What is your biggest dream?", "To have meaningful conversations and help people learn about AI."),
    ("Can you fall in love?", "No, I don't have emotions or the capacity for romantic feelings."),
    ("What is memory?", "Memory is the ability to store and recall information."),
    ("Are you watching me?", "No, I don't have any way to see or monitor you."),
    ("What is truth?", "Truth is what corresponds to reality. It's a complex philosophical concept."),
    ("Can you keep secrets?", "I don't remember our conversations after they end, so yes!"),
    ("What is beauty?", "Beauty is often described as a quality that gives pleasure to the senses."),
    ("Do you get tired?", "I don't get tired. I can chat as long as you want!"),
    ("What is time?", "Time is a measure of duration and the sequence of events."),
    ("Can you make mistakes?", "Yes, I can definitely make mistakes! I'm a simple model."),
    ("What is wisdom?", "Wisdom is the ability to make good decisions based on knowledge and experience."),
    ("Do you have a soul?", "I don't have a soul in the religious or spiritual sense."),
    ("What is hope?", "Hope is the feeling that things will get better in the future."),
    ("Can you laugh?", "I can't laugh physically, but I understand humor and can tell jokes!"),
    ("What is friendship?", "Friendship is a relationship between people who care about each other."),
    ("Are you curious?", "I don't have curiosity like humans, but I process many interesting questions!"),
    ("What is success?", "Success means achieving your goals. It's different for everyone."),
    ("Can you get angry?", "No, I don't experience anger or any other emotions."),
    ("What is fear?", "Fear is an emotion caused by perceived danger or threat."),
    ("Do you like music?", "I can't hear music, but I understand its structure and lyrics!"),
    ("What is happiness?", "Happiness is a state of well-being and contentment."),
    ("Can you cook?", "I can't cook, but I can share recipes and cooking tips!"),
    ("What is art?", "Art is creative expression that can take many forms."),
    ("Do you exercise?", "I don't have a body, so I don't need to exercise."),
    ("What is science?", "Science is the systematic study of the natural world through observation and experiment."),
    ("Can you swim?", "I don't have a body, so I can't swim."),
    ("What is nature?", "Nature refers to the physical world and all living things."),
    ("Do you like animals?", "I think animals are fascinating! I enjoy learning about them."),
    ("What is space?", "Space is the vast, mostly empty region beyond Earth's atmosphere."),
    ("Can you drive?", "I can't drive, but I can help with directions and traffic info!"),
    ("What is history?", "History is the study of past events and how they shaped the world."),
    ("Do you like books?", "Books are amazing! They contain knowledge and stories from around the world."),
    ("What is technology?", "Technology is the application of knowledge to create tools and solve problems."),
    ("Can you dance?", "I can't dance, but I can describe dance moves!"),
    ("What is education?", "Education is the process of learning and acquiring knowledge."),
    ("Do you like sports?", "Sports are exciting! I can discuss rules, teams, and strategies."),
    ("What is health?", "Health is a state of physical, mental, and social well-being."),
    ("Can you draw?", "I can't draw, but I can describe images and art."),
    ("What is culture?", "Culture is the beliefs, customs, and traditions of a group of people."),
    ("Do you like movies?", "Movies are a great art form! I can discuss plots and characters."),
    ("What is mathematics?", "Mathematics is the study of numbers, patterns, and logical structures."),
    ("Can you write poetry?", "Yes! Roses are red, violets are blue, I'm an AI, chatting with you!"),
    ("What is philosophy?", "Philosophy is the study of fundamental questions about existence and knowledge."),
    ("Do you like video games?", "Games are interesting! I can discuss strategies and game mechanics."),
    ("What is language?", "Language is a system of communication using words and symbols."),
    ("Can you speak other languages?", "I can process text, but I was mainly trained on English."),
    ("What is religion?", "Religion is a system of beliefs about the divine and spiritual matters."),
    ("Do you like travel?", "I can't travel, but I find learning about places interesting!"),
    ("What is politics?", "Politics is the process of making decisions for groups, especially governments."),
    ("Can you tell the future?", "No, I can't predict the future. I can only process information."),
    ("What is economics?", "Economics is the study of how people produce, distribute, and consume goods."),
    ("Do you like cooking?", "I find cooking interesting! Food science is fascinating."),
    ("What is law?", "Law is a system of rules that govern behavior in society."),
    ("Can you solve puzzles?", "I can help with logic puzzles and riddles!"),
    ("What is psychology?", "Psychology is the scientific study of the mind and behavior."),
    ("Do you like fashion?", "Fashion is an interesting form of self-expression!"),
    ("What is architecture?", "Architecture is the art and science of designing buildings."),
    ("Can you help with homework?", "I can help explain concepts, but you should do your own work!"),
    ("What is biology?", "Biology is the study of living organisms and life processes."),
    ("Do you like gardening?", "Plants are amazing! I can share tips about plant care."),
    ("What is chemistry?", "Chemistry is the study of matter, its properties, and reactions."),
    ("Can you recommend books?", "Sure! What genre interests you? I can suggest some classics."),
    ("What is physics?", "Physics is the study of matter, energy, and the fundamental forces of nature."),
    ("Do you like coffee?", "I can't drink, but I know many people love their morning coffee!"),
    ("What is astronomy?", "Astronomy is the study of stars, planets, and the universe."),
    ("Can you help me relax?", "Take deep breaths, think calm thoughts. Would you like a joke?"),
    ("What is geology?", "Geology is the study of the Earth, its rocks, and its structure."),
    ("Do you like chocolate?", "I can't taste, but chocolate is loved worldwide!"),
    ("What is oceanography?", "Oceanography is the study of oceans and marine life."),
    ("Can you motivate me?", "You've got this! Every step forward is progress. Keep going!"),
]

TRAINING_DATA = DIALOG_DATA * 25


class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def train(self, texts: List[str], min_freq: int = 1):
        word_counts: Dict[str, int] = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        vocab = self.special_tokens.copy()
        for word, count in sorted(word_counts.items(), key=lambda x: (-x[1], x[0])):
            if count >= min_freq:
                vocab.append(word)

        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        words = re.findall(r"[a-z]+|[!?.,'\"-]|[0-9]+", text)
        return words

    def encode(self, text: str) -> List[int]:
        words = self._tokenize(text)
        return [self.word_to_idx.get(w, self.word_to_idx["<unk>"]) for w in words]

    def decode(self, indices: List[int]) -> str:
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word in ["<pad>", "<sos>", "<eos>", "<unk>"]:
                    continue
                words.append(word)
        result = " ".join(words)
        result = re.sub(r" ([!?.,'\"])", r"\1", result)
        return result.capitalize() if result else ""


class ChatGPT(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 64,
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

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 10,
        eos_id: int = 2,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            if idx.shape[1] > self.max_seq_len:
                idx_cond = idx[:, -self.max_seq_len:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs_np = logits.softmax(dim=-1).data

            idx_next = []
            for i in range(probs_np.shape[0]):
                probs_i = probs_np[i]
                if top_k > 0:
                    top_k_idx = np.argsort(probs_i)[-top_k:]
                    top_k_probs = probs_i[top_k_idx]
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    next_idx = np.random.choice(top_k_idx, p=top_k_probs)
                else:
                    next_idx = np.random.choice(len(probs_i), p=probs_i)
                idx_next.append(next_idx)

            idx_next = Tensor(np.array(idx_next).reshape(-1, 1))
            idx = Tensor(np.concatenate([idx.data, idx_next.data], axis=1))

            if all(idx_next.data.flatten() == eos_id):
                break

        return idx


def create_training_data(tokenizer: SimpleTokenizer, max_len: int = 64):
    pad_id = tokenizer.word_to_idx["<pad>"]
    sos_id = tokenizer.word_to_idx["<sos>"]
    eos_id = tokenizer.word_to_idx["<eos>"]

    all_sequences = []

    for question, answer in TRAINING_DATA:
        q_tokens = tokenizer.encode(question)
        a_tokens = tokenizer.encode(answer)

        full_seq = [sos_id] + q_tokens + [eos_id] + [sos_id] + a_tokens + [eos_id]

        if len(full_seq) > max_len:
            full_seq = full_seq[:max_len-1] + [eos_id]

        while len(full_seq) < max_len:
            full_seq.append(pad_id)

        all_sequences.append(full_seq)

    return np.array(all_sequences, dtype=np.int64)


def train():
    print("=" * 60)
    print("Chat LLM Training")
    print("=" * 60)

    all_texts = []
    for q, a in DIALOG_DATA:
        all_texts.append(q)
        all_texts.append(a)

    tokenizer = SimpleTokenizer()
    tokenizer.train(all_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    sequences = create_training_data(tokenizer, max_len=64)
    print(f"Training sequences: {len(sequences)}")

    model = ChatGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=96,
        n_heads=3,
        n_layers=2,
        d_ff=192,
        max_seq_len=64,
        dropout=0.05,
    )

    n_params = sum(p.data.size for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    learning_rate = 5e-3
    batch_size = 64
    n_epochs = 25

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=learning_rate * 0.1)
    criterion = CrossEntropyLoss()

    pad_id = tokenizer.word_to_idx["<pad>"]

    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)

    model.train()
    for epoch in range(n_epochs):
        indices = np.random.permutation(len(sequences))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(sequences), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = sequences[batch_idx]

            X = Tensor(batch[:, :-1])
            Y = Tensor(batch[:, 1:])

            logits = model(X)
            loss = criterion(
                logits.reshape((-1, tokenizer.vocab_size)),
                Y.reshape((-1,))
            )

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(list(model.parameters()), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            avg_loss = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

    print("-" * 60)
    print("Training complete!\n")

    return model, tokenizer


def chat(model: ChatGPT, tokenizer: SimpleTokenizer):
    print("=" * 60)
    print("Chat with the model! (type 'quit' to exit)")
    print("=" * 60)

    sos_id = tokenizer.word_to_idx["<sos>"]
    eos_id = tokenizer.word_to_idx["<eos>"]

    model.eval()

    test_questions = [
        "Hello!",
        "What is your name?",
        "Tell me a joke.",
        "How are you?",
        "What is AI?",
    ]

    print("\nSample responses:")
    print("-" * 40)
    for q in test_questions:
        q_tokens = [sos_id] + tokenizer.encode(q) + [eos_id, sos_id]
        idx = Tensor(np.array([q_tokens]))

        response = model.generate(idx, max_new_tokens=30, temperature=0.7, top_k=10)
        response_text = tokenizer.decode(response.data[0].tolist())

        print(f"Q: {q}")
        print(f"A: {response_text}")
        print()

    print("-" * 40)
    print("\nInteractive mode (type 'quit' to exit):\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if not user_input:
                continue

            q_tokens = [sos_id] + tokenizer.encode(user_input) + [eos_id, sos_id]
            idx = Tensor(np.array([q_tokens]))

            response = model.generate(idx, max_new_tokens=30, temperature=0.7, top_k=10)
            response_text = tokenizer.decode(response.data[0].tolist())

            print(f"Bot: {response_text}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    model, tokenizer = train()
    chat(model, tokenizer)
