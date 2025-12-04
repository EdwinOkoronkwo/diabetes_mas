import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os

# --- Constants ---
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 128
MODEL_DIR = "models"

# --- Model Definition (Must match MLModelWrapper.py) ---
class DiabetesRiskTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, text_indices):
        embedded = self.embedding(text_indices)
        embedded = embedded.permute(0, 2, 1)
        pooled = self.pool(embedded).squeeze(2)
        return self.fc(pooled)

# --- Helper Functions ---
def text_to_indices(text, word_to_idx, max_len=MAX_SEQUENCE_LENGTH):
    indices = [word_to_idx.get(word, 0) for word in text.split()]
    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))
    return torch.tensor(indices[:max_len], dtype=torch.long)

def train():
    print("Starting training...")
    
    # 1. Dummy Data
    data = [
        ("blurred vision and fatigue", 1),
        ("frequent urination and thirst", 1),
        ("unexplained weight loss", 1),
        ("feeling very tired and hungry", 1),
        ("slow healing sores", 1),
        ("feeling great", 0),
        ("normal checkup results", 0),
        ("no symptoms reported", 0),
        ("energy levels are good", 0),
        ("sleeping well", 0)
    ]
    
    # 2. Build Vocabulary
    vocab = set()
    for text, _ in data:
        for word in text.split():
            vocab.add(word)
            
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)} # 0 is reserved for padding/unknown
    word_to_idx["<UNK>"] = 0
    vocab_size = len(word_to_idx) + 1
    
    print(f"Vocabulary size: {vocab_size}")
    
    # 3. Prepare Data
    X = torch.stack([text_to_indices(text, word_to_idx) for text, _ in data])
    y = torch.tensor([label for _, label in data], dtype=torch.long)
    
    # 4. Initialize Model
    model = DiabetesRiskTextClassifier(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 5. Train Loop
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    # 6. Save Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    model_path = os.path.join(MODEL_DIR, "risk_model.pth")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.joblib")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(word_to_idx, tokenizer_path)
    
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    train()
