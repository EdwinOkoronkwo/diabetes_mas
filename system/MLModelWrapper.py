import torch
import torch.nn as nn
import numpy as np
import joblib
import os  # Imported for robust path handling
from typing import List

# --- Constants and Architecture copied from train_risk_model.py ---

# Robust Path Resolution:
# This ensures the script can find the model files regardless of the current working directory.
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming the 'models' directory is one level up and then in 'models/'
# E.g., if this file is in 'agents/', it looks in '../models/'
MODEL_FILE_DIR = os.path.join(CURRENT_FILE_DIR, '..', 'models')

MODEL_STATE_DICT_PATH = os.path.join(MODEL_FILE_DIR, "risk_model.pth")
TOKENIZER_PATH = os.path.join(MODEL_FILE_DIR, "tokenizer.joblib")

import torch
import torch.nn as nn

# Define a constant for sequence length to be used consistently
# Sequence length is 20 to prevent symptom truncation
MAX_SEQUENCE_LENGTH = 20
# FIX: Increased embedding dimension from 50 to 128 to give the model more capacity
# to learn the severity of high-risk keywords like 'blurred vision'.
EMBEDDING_DIM = 128


# 1. Model Definition (Architecture)
class DiabetesRiskTextClassifier(nn.Module):
    """A simple PyTorch text classifier using an Embedding layer and linear output."""

    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        # Use pooling to aggregate embeddings across the sequence length
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, text_indices):
        # text_indices: [batch_size, sequence_length]
        embedded = self.embedding(text_indices)
        # Permute to (batch_size, embedding_dim, seq_len) for pooling
        embedded = embedded.permute(0, 2, 1)
        pooled = self.pool(embedded).squeeze(2)  # [batch_size, embedding_dim]
        return self.fc(pooled)  # [batch_size, num_classes]


# 2. Tokenization Helper (Preprocessing Logic)
def text_to_indices(text, word_to_idx, max_len=MAX_SEQUENCE_LENGTH):
    """Converts a text string to a padded/truncated tensor of indices using the vocabulary map."""

    # Get index for each word, using 0 (UNKNOWN/Padding) as the default
    indices = [word_to_idx.get(word, 0) for word in text.split()]

    # Simple padding/truncation
    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))

    return torch.tensor(indices[:max_len], dtype=torch.long)


# ----------------------------------------------------------------------


class MLModelWrapper:
    """
    Wrapper for loading and running the PyTorch text classification model.
    It handles loading the architecture, state dictionary (.pth), and tokenizer (.joblib).
    """

    def __init__(self):
        self.model = None
        self.word_to_idx = None

        self.device = torch.device("cpu")
        print(f"[DEBUG: Init] Using device: {self.device}")

        # DEBUG: Report the absolute paths being used
        print(f"[DEBUG: Init] Tokenizer path being used: {TOKENIZER_PATH}")
        print(f"[DEBUG: Init] Model path being used: {MODEL_STATE_DICT_PATH}")

        print(f"[MLModelWrapper] Attempting to load tokenizer from {TOKENIZER_PATH}...")
        try:
            # 1. Load the vocabulary (tokenizer)
            self.word_to_idx = joblib.load(TOKENIZER_PATH)
            # The vocabulary size needs to account for the UNK token (index 0)
            vocab_size = len(self.word_to_idx) + 1
            print(f"[DEBUG: Init] Tokenizer loaded. Vocabulary size: {vocab_size}")

        except FileNotFoundError:
            # Note: The path printed in the error is now absolute, making it easier to check manually.
            raise FileNotFoundError(
                f"CRITICAL ERROR: Tokenizer file not found at: {TOKENIZER_PATH}. Train the model first!")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer vocabulary: {e}")

        print(f"[MLModelWrapper] Attempting to load PyTorch model from {MODEL_STATE_DICT_PATH}...")
        try:
            # 2. Reconstruct the model architecture (Hyperparameters must match training)
            EMBEDDING_DIM = 50
            NUM_CLASSES = 2

            print(
                f"[DEBUG: Init] Model params: EMBEDDING_DIM={EMBEDDING_DIM}, NUM_CLASSES={NUM_CLASSES}, MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}")

            self.model = DiabetesRiskTextClassifier(
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                num_classes=NUM_CLASSES
            ).to(self.device)

            # 3. Load the saved weights (state dictionary)
            state_dict = torch.load(MODEL_STATE_DICT_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set model to evaluation mode

            print("[MLModelWrapper] PyTorch Model and Tokenizer loaded successfully.")

        except FileNotFoundError:
            # Note: The path printed in the error is now absolute, making it easier to check manually.
            raise FileNotFoundError(
                f"CRITICAL ERROR: Model state dict not found at: {MODEL_STATE_DICT_PATH}. Train the model first!")
        except Exception as e:
            # Catch all other loading/unpickling errors
            raise RuntimeError(f"Failed to load PyTorch model weights: {e}")

    def predict_proba(self, text_list: List[str]) -> np.ndarray:
        """
        Predicts the probability of the positive class (risk=1) using the PyTorch model.
        The input is expected to be a list of symptom strings.
        """
        if self.model is None or self.word_to_idx is None:
            raise ValueError("Model or tokenizer not initialized. Cannot predict.")

        # Ensure input is always handled as a list
        if not isinstance(text_list, list):
            text_list = [text_list]

        print(f"[DEBUG: Predict] Received {len(text_list)} inputs for prediction.")

        # 1. Tokenize input text list
        input_tensors = [
            text_to_indices(text, self.word_to_idx) for text in text_list
        ]

        # 2. Stack tensors into a single batch and move to device
        input_batch = torch.stack(input_tensors).to(self.device)

        print(f"[DEBUG: Predict] Input batch shape (BatchSize, SeqLen): {input_batch.shape}")

        # 3. Run prediction (forward pass)
        with torch.no_grad():
            outputs = self.model(input_batch)

        print(f"[DEBUG: Predict] Output logits shape (BatchSize, NumClasses): {outputs.shape}")

        # 4. Apply Softmax to convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        # 5. Extract the probability of the positive class (index 1)
        return probabilities[:, 1]




# import torch
# import torch.nn as nn
# import numpy as np
# import joblib
# import os
# from typing import List
#
# MODEL_PATH = "models/risk_model.pth"
# TOKENIZER_PATH = "models/tokenizer.joblib"
#
# # --- Constants and Architecture copied from train_risk_model.py ---
# class TorchModelWrapper:
#     """
#     Loads the PyTorch diabetes classifier + tokenizer.
#     Reconstructs the architecture EXACTLY as trained.
#     """
#
#     def __init__(self, model_path, tokenizer_path, embedding_dim=50, num_classes=2, max_len=5):
#         import torch
#         import torch.nn as nn
#         from joblib import load
#         import logging
#
#         self.logger = logging.getLogger(__name__)
#         self.max_len = max_len
#         self.embedding_dim = embedding_dim
#         self.num_classes = num_classes
#
#         # ---- Load tokenizer (word_to_idx) ----
#         try:
#             self.word_to_idx = load(tokenizer_path)
#             self.logger.info(f"[TorchModelWrapper] Loaded tokenizer from {tokenizer_path}")
#         except Exception as e:
#             raise RuntimeError(f"Failed to load tokenizer: {e}")
#
#         vocab_size = len(self.word_to_idx) + 1
#
#         # ---- Rebuild model architecture ----
#         class DiabetesRiskTextClassifier(nn.Module):
#             def __init__(self, vocab_size, embedding_dim, num_classes):
#                 super().__init__()
#                 self.embedding = nn.Embedding(vocab_size, embedding_dim)
#                 self.fc = nn.Linear(embedding_dim, num_classes)
#                 self.pool = nn.AdaptiveAvgPool1d(1)
#
#             def forward(self, text_indices):
#                 embedded = self.embedding(text_indices)
#                 embedded = embedded.permute(0, 2, 1)
#                 pooled = self.pool(embedded).squeeze(2)
#                 return self.fc(pooled)
#
#         self.model = DiabetesRiskTextClassifier(
#             vocab_size=vocab_size,
#             embedding_dim=embedding_dim,
#             num_classes=num_classes
#         )
#
#         # ---- Load state_dict ----
#         try:
#             state = torch.load(model_path, map_location="cpu")
#             self.model.load_state_dict(state)
#             self.model.eval()
#             self.logger.info(f"[TorchModelWrapper] Loaded PyTorch model from {model_path}")
#         except Exception as e:
#             raise RuntimeError(f"Failed loading model weights: {e}")
#
#     # ------------------------
#     # Convert text → indices
#     # ------------------------
#     def text_to_indices(self, text):
#         words = text.split()
#         idxs = [self.word_to_idx.get(w, 0) for w in words]
#
#         # Pad/truncate
#         if len(idxs) < self.max_len:
#             idxs.extend([0] * (self.max_len - len(idxs)))
#         idxs = idxs[:self.max_len]
#
#         import torch
#         return torch.tensor(idxs, dtype=torch.long)
#
#     # ------------------------
#     # Prediction
#     # ------------------------
#     def predict_proba(self, text_list):
#         import torch
#         import torch.nn.functional as F
#
#         if not isinstance(text_list, list):
#             text_list = [text_list]
#
#         batch = torch.stack([self.text_to_indices(t) for t in text_list])
#         logits = self.model(batch)
#         probs = F.softmax(logits, dim=1)
#
#         # Return probability of class 1
#         return probs[:, 1].detach().numpy()
