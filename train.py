import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import AutoregressiveGenerator, SimpleCNNClassifier

def train_classifier():
    """Train a simple classifier for reward computation"""
    transform = transforms.Compose([
        transforms.Resize(8),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    classifier = SimpleCNNClassifier()
    optimizer = torch.optim.Adam(classifier.parameters())

    print("Training classifier...")
    for epoch in range(3):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    torch.save(classifier.state_dict(), 'classifier.pth')
    return classifier

def compute_reward(images, target_class, classifier):
    """
    Compute reward for generated images

    Args:
        images: (batch, 64) pixel values
        target_class: Target digit class
        classifier: Trained classifier

    Returns:
        rewards: (batch,) reward values
    """
    # Reshape to 8x8 and normalize
    imgs = images.float().view(-1, 1, 8, 8) / 255.0

    with torch.no_grad():
        logits = classifier(imgs)
        probs = F.softmax(logits, dim=-1)
        target_probs = probs[:, target_class]

    # Reward is confidence in target class
    rewards = target_probs

    return rewards

def train_generator(epochs=100, batch_size=16):
    """Train autoregressive generator with policy gradients"""

    # Load or train classifier
    try:
        classifier = SimpleCNNClassifier()
        classifier.load_state_dict(torch.load('classifier.pth'))
        classifier.eval()
    except:
        classifier = train_classifier()
        classifier.eval()

    # Create generator
    generator = AutoregressiveGenerator(hidden_dim=128)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    target_digit = 7  # Try to generate digit 7

    rewards_history = []

    print(f"\nTraining generator to create digit {target_digit}...")
    print("-" * 60)

    for epoch in range(epochs):
        # Generate images
        images = generator.generate(batch_size=batch_size, seq_len=64, temperature=1.0)

        # Compute rewards
        rewards = compute_reward(images, target_digit, classifier)
        rewards_history.append(rewards.mean().item())

        # Compute loss (REINFORCE algorithm)
        # We need log probabilities of generated sequence
        logits, _ = generator(images[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of chosen actions
        chosen_log_probs = log_probs.gather(2, images[:, 1:].unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_log_probs.sum(dim=1)  # Sum over sequence

        # Policy gradient loss
        loss = -(chosen_log_probs * rewards).mean()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Reward = {rewards.mean().item():.4f}, "
                  f"Loss = {loss.item():.4f}")

            # Visualize samples
            if (epoch + 1) % 50 == 0:
                visualize_samples(generator, epoch + 1, target_digit)

    # Save model
    torch.save(generator.state_dict(), 'generator.pth')

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('training_progress.png')
    print("\nTraining complete!")

def visualize_samples(generator, epoch, target_digit):
    """Visualize generated samples"""
    with torch.no_grad():
        images = generator.generate(batch_size=16, seq_len=64, temperature=0.8)
        images = images.cpu().numpy().reshape(-1, 8, 8)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'Epoch {epoch} - Generating Digit {target_digit}')

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'samples_epoch_{epoch}.png')
    plt.close()

if __name__ == "__main__":
    train_generator(epochs=100, batch_size=16)