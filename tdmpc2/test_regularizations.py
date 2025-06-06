# test_regularizations.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functorch import jacrev
import matplotlib.pyplot as plt
import numpy as np
import time

# Import our regularizations
from regularizations import orthogonality_regularization, full_rank_regularization

# ----------------------------------------------------------------------
# 1. Hyper-parameters
# ----------------------------------------------------------------------
in_dim = 28 * 28          # MNIST flattened
encoder_dim = 256         # Output dimension for encoder (required by regularizations)
hidden_dims = [128]  # Hidden layers
num_classes = 10          # MNIST classes
batch_size = 64           # Smaller batch for efficiency with regularizations
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Regularization weights
ortho_weight = 0.01       # Weight for orthogonality regularization
rank_weight = 0.01        # Weight for full-rank regularization

print(f"Using device: {device}")

# ----------------------------------------------------------------------
# 2. Model: Encoder + Classifier
# ----------------------------------------------------------------------
class EncoderClassifier(nn.Module):
    """
    A model with an encoder that outputs 512-D features,
    followed by a classifier head.
    """
    def __init__(self, in_dim, encoder_dim, hidden_dims, num_classes):
        super().__init__()
        
        # Encoder: input -> 512-D features
        encoder_layers = []
        dims = [in_dim] + hidden_dims + [encoder_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on last encoder layer
                encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier: 512-D features -> num_classes
        self.classifier = nn.Linear(encoder_dim, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def encode(self, x):
        """Get encoded features only"""
        return self.encoder(x)

# ----------------------------------------------------------------------
# 3. Data
# ----------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten
])

train_set = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# ----------------------------------------------------------------------
# 5. Training with and without regularizations
# ----------------------------------------------------------------------
def train_model(use_ortho=True, use_rank=True, epochs=10):
    """Train model with or without regularizations"""
    
    model = EncoderClassifier(in_dim, encoder_dim, hidden_dims, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    ortho_vals = []
    rank_vals = []
       
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_ortho = 0.0
        epoch_rank = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y)
            
            # Compute regularization losses
            if use_ortho:
                # Compute with gradients for optimization
                ortho_loss = orthogonality_regularization(
                    model.encoder, x, 
                    num_pairs=4, hutchinson_samples=1, device=device, latent_dim=encoder_dim
                )
            else:
                if batch_idx % 200 == 0:
                    # Compute without gradients for monitoring only
                    with torch.no_grad():
                        ortho_loss = orthogonality_regularization(
                            model.encoder, x, 
                            num_pairs=4, hutchinson_samples=1, device=device, latent_dim=encoder_dim
                        )
                else:
                    ortho_loss = torch.tensor(0.0, device=device)

            if use_rank:
                # Compute with gradients for optimization
                rank_loss, abs_dets = full_rank_regularization(
                    model.encoder, x,
                    epsilon=1e-4, activation_margin=10.0, device=device, latent_dim=encoder_dim
                )
            else:
                if batch_idx % 200 == 0:
                    # Compute without gradients for monitoring only
                    with torch.no_grad():
                        rank_loss, abs_dets = full_rank_regularization(
                            model.encoder, x,
                            epsilon=1e-4, activation_margin=10.0, device=device, latent_dim=encoder_dim
                        )
                else:
                    rank_loss = torch.tensor(0.0, device=device)
                    abs_dets = torch.tensor(0.0, device=device)


            abs_det = abs_dets.mean()
            
            # Total loss
            total_loss_tensor = ce_loss
            if use_ortho:
                total_loss_tensor += ortho_weight * ortho_loss
            if use_rank:
                total_loss_tensor += rank_weight * rank_loss
            
            # Backward pass
            
            total_loss_tensor.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Statistics
            total_loss += total_loss_tensor.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
            
            epoch_ortho += ortho_loss.item()
            epoch_rank += rank_loss.item()
            
            if batch_idx % 200 == 0:
                # Compute and print the ortho loss and determinant
                print(f"    Batch {batch_idx}:")
                print(f"      Ortho loss: {ortho_loss.item():.6f} {'(used)' if use_ortho else '(not used)'}")
                print(f"      Rank loss: {rank_loss.item():.6f}, Determinant (mean): {abs_det.item():.6f} {'(used)' if use_rank else '(not used)'}")
        
        # Compute epoch statistics
        avg_loss = total_loss / len(train_loader)
        train_acc = (total_correct / total_samples) * 100
        avg_ortho = epoch_ortho / len(train_loader)
        avg_rank = epoch_rank / len(train_loader)
        
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        ortho_vals.append(avg_ortho)
        rank_vals.append(avg_rank)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Avg Ortho: {avg_ortho:.6f}, Avg Rank: {avg_rank:.6f}")
            
    return model, train_losses, train_accs, ortho_vals, rank_vals

# ----------------------------------------------------------------------
# 6. Main experiment
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # Train without regularizations
    print("Training baseline model (no regularizations)...")
    model_baseline, losses_base, accs_base, ortho_base, rank_base = train_model(
        use_ortho=False, use_rank=False, epochs=8
    )

        # Train with regularizations  
    print("\nTraining model with regularizations...")
    model_reg, losses_reg, accs_reg, ortho_reg, rank_reg = train_model(
        use_ortho=True, use_rank=True, epochs=8
    )
    
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Final training accuracy:")
    print(f"  Baseline: {accs_base[-1]:.2f}%")
    print(f"  Regularized: {accs_reg[-1]:.2f}%")