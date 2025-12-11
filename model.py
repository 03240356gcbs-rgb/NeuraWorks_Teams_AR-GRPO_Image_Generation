import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Autoregressive Pixel-by-Pixel Generator
# ---------------------------------------------------------
class AutoregressiveGenerator(nn.Module):
    """
    Simplified autoregressive image generator
    Generates 8x8 images pixel by pixel (64 pixels)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        # Embedding for pixel values 0–255
        self.embedding = nn.Embedding(256, hidden_dim)

        # LSTM for sequential generation
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Output layer predicts next pixel (0–255)
        self.output = nn.Linear(hidden_dim, 256)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len) tensor of previous pixels
            hidden: previous LSTM hidden state

        Returns:
            logits: (batch, seq_len, 256)
            hidden: updated hidden state
        """
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        logits = self.output(lstm_output)
        return logits, hidden

    def generate(self, batch_size=1, seq_len=64, temperature=1.0):
        """
        Autoregressive image generator.

        Args:
            batch_size: number of images to generate
            seq_len: number of pixels (64 = 8x8)
            temperature: sampling temperature

        Returns:
            images: (batch_size, seq_len) tensor of pixel values
        """
        device = next(self.parameters()).device
        images = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        hidden = None

        for t in range(seq_len):
            if t == 0:
                x = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            else:
                x = images[:, t-1:t]

            logits, hidden = self.forward(x, hidden)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_pixel = torch.multinomial(probs, 1).squeeze(-1)

            images[:, t] = next_pixel

        return images


# ---------------------------------------------------------
# Simple CNN Classifier for Reward Calculation
# ---------------------------------------------------------
class SimpleCNNClassifier(nn.Module):
    """
    Simple CNN classifier for computing rewards
    from generated 8x8 MNIST-like images.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 8×8 → 8×8
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 8×8 → 8×8
        self.pool = nn.MaxPool2d(2)  # 8×8 → 4×4 → 2×2

        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 8, 8)

        Returns:
            logits: (batch, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))  # 8→4
        x = self.pool(F.relu(self.conv2(x)))  # 4→2

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
