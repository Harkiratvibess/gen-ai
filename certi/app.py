import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# -----------------------------------------------------------
#                    TEXT PREPARATION
# -----------------------------------------------------------

def prepare_text(text):
    """Returns vocab, mappings, cleaned dataset."""
    text = text.replace("\r", "")
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return text, chars, char_to_idx, idx_to_char, len(chars)

# -----------------------------------------------------------
#                    CHARACTER DATASET
# -----------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, text, seq_len, char_to_idx):
        self.text = text
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        seq = self.text[idx:idx + self.seq_len]
        target = self.text[idx + 1:idx + self.seq_len + 1]

        seq_idx = torch.tensor([self.char_to_idx[ch] for ch in seq], dtype=torch.long)
        target_idx = torch.tensor([self.char_to_idx[ch] for ch in target], dtype=torch.long)
        return seq_idx, target_idx

# -----------------------------------------------------------
#            GRU MODEL (better than RNN)
# -----------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# -----------------------------------------------------------
#                 TEXT GENERATION
# -----------------------------------------------------------

def generate_text(model, seed, char_to_idx, idx_to_char, seq_length, length, temperature, device):
    model.eval()

    if any(ch not in char_to_idx for ch in seed):
        raise ValueError("Seed contains unseen characters.")

    seed = seed[-seq_length:]
    while len(seed) < seq_length:
        seed = seed[0] + seed

    input_seq = torch.tensor([char_to_idx[ch] for ch in seed], dtype=torch.long).unsqueeze(0).to(device)

    hidden = None
    generated = seed

    for _ in range(length):
        logits, hidden = model(input_seq, hidden)
        hidden = hidden.detach()

        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        next_char = idx_to_char[next_idx]
        generated += next_char

        input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return generated

# -----------------------------------------------------------
#                       STREAMLIT UI
# -----------------------------------------------------------

st.set_page_config(page_title="Custom Gen-AI Model", layout="wide")

st.title("🧠 Build & Train Your Own Gen-AI Model (Character Level)")

with st.sidebar:
    st.header("Model Settings")

    embed_size = st.slider("Embedding Size", 8, 256, 64)
    hidden_size = st.slider("Hidden Size", 32, 512, 128)
    num_layers = st.slider("GRU Layers", 1, 4, 2)
    seq_length = st.slider("Sequence Length", 10, 200, 60)
    batch_size = st.slider("Batch Size", 8, 128, 32)
    epochs = st.slider("Epochs", 1, 50, 5)
    lr = st.slider("Learning Rate", 0.0001, 0.01, 0.002, step=0.0001)
    device_choice = st.radio("Device", ["cpu", "cuda"], index=0)

    device = "cuda" if device_choice == "cuda" and torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
#           LOAD TRAINING TEXT
# -----------------------------------------------------------

st.subheader("📄 Training Text")
text = st.text_area("Paste your training text here:", height=200)

if not text.strip():
    st.warning("Please enter some training text to begin.")
    st.stop()

# prep
text, chars, char_to_idx, idx_to_char, vocab_size = prepare_text(text)

# -----------------------------------------------------------
#                SESSION STATE MODEL
# -----------------------------------------------------------

if "model" not in st.session_state:
    st.session_state.model = GRUModel(vocab_size, embed_size, hidden_size, num_layers).to(device)

model = st.session_state.model

# -----------------------------------------------------------
#                TRAINING BUTTON
# -----------------------------------------------------------

if st.button("🚀 Train Model"):
    dataset = CharDataset(text, seq_length, char_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    progress = st.progress(0)
    loss_area = st.empty()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        progress.progress((epoch + 1) / epochs)
        loss_area.write(f"**Epoch {epoch+1}/{epochs} — Loss:** {avg_loss:.4f}")

    st.success("Training complete!")

# -----------------------------------------------------------
#                TEXT GENERATION
# -----------------------------------------------------------

st.subheader("🔮 Text Generation")

seed = st.text_input("Seed text:", "hello")
gen_length = st.slider("Generated Length", 50, 1000, 200)
temperature = st.slider("Temperature", 0.1, 2.0, 0.8)

if st.button("✨ Generate"):
    try:
        output = generate_text(
            model, seed, char_to_idx, idx_to_char,
            seq_length, gen_length, temperature, device
        )
        st.text_area("Generated Output:", output, height=300)
    except Exception as e:
        st.error(str(e))
