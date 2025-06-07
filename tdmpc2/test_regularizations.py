# test_regularizations.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import our regularizations
from regularizations import orthogonality_regularization, full_rank_regularization


# ----------------------------------------------------------------------
# 1. Hyper-parameters
# ----------------------------------------------------------------------
in_dim        = 28 * 28            # MNIST flattened
encoder_dim   = 256                # Encoder output dimension
hidden_dims   = [128, 128]         # Hidden layers
num_classes   = 10                 # MNIST classes
batch_size    = 64                 # Smaller batch for efficiency
lr            = 1e-3
device        = "cuda" if torch.cuda.is_available() else "cpu"

# Regularization weights
ortho_weight  = 0.01
rank_weight   = 0.01

print(f"Using device: {device}")

# ----------------------------------------------------------------------
# 2. Model: Encoder + Classifier
# ----------------------------------------------------------------------
class EncoderClassifier(nn.Module):
    """
    A model with an encoder that outputs `encoder_dim`-D features,
    followed by a classifier head.
    """
    def __init__(self, in_dim, encoder_dim, hidden_dims, num_classes):
        super().__init__()

        # Encoder
        encoder_layers = []
        dims = [in_dim] + hidden_dims + [encoder_dim]
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:           # no activation on last layer
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Classifier head
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits   = self.classifier(features)
        return logits

    def encode(self, x):
        return self.encoder(x)

# ----------------------------------------------------------------------
# 3. Data
# ----------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.view(-1))  # flatten to vector
])

train_set = datasets.MNIST(root=".", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

# ----------------------------------------------------------------------
# 4. Utility: global L2-norm of a list of tensors
# ----------------------------------------------------------------------
def global_grad_norm(grads):
    """Return (∑‖g‖₂²)^{1/2}; treat None as zero."""
    total = torch.tensor(0.0, device=device)
    for g in grads:
        if g is not None:
            total += g.norm() ** 2
    return total.sqrt().item()

# ----------------------------------------------------------------------
# 5. Training function
# ----------------------------------------------------------------------
def train_model(use_ortho=True, use_rank=True, epochs=10):
    model = EncoderClassifier(in_dim, encoder_dim, hidden_dims, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        epoch_ortho, epoch_rank, ortho_count, rank_count = 0.0, 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits    = model(x)
            ce_loss   = F.cross_entropy(logits, y)

            # Regularisation losses
            if use_ortho:
                ortho_loss = orthogonality_regularization(
                    model.encoder, x, device=device, latent_dim=encoder_dim
                )
                ortho_count += 1
            else:
                ortho_loss = torch.tensor(0.0, device=device)

            if use_rank:
                rank_loss, abs_dets = full_rank_regularization(
                    model.encoder, x,
                    epsilon=1e-4, activation_margin=5.0,
                    device=device
                )
                rank_count += 1
            else:
                rank_loss  = torch.tensor(0.0, device=device)


            # ----------------------------------------------------------
            # Gradient norms of *individual* losses (before they mix)
            # ----------------------------------------------------------
            ce_grads = torch.autograd.grad(
                ce_loss, model.parameters(), retain_graph=True, allow_unused=True
            )
            ce_grad_norm = global_grad_norm(ce_grads)

            if use_ortho:
                ortho_grads = torch.autograd.grad(
                    ortho_loss, model.parameters(), retain_graph=True, allow_unused=True
                )
                ortho_grad_norm = global_grad_norm(ortho_grads)
            else:
                ortho_grad_norm = 0.0

            if use_rank:
                rank_grads = torch.autograd.grad(
                    rank_loss, model.parameters(), retain_graph=True, allow_unused=True
                )
                rank_grad_norm = global_grad_norm(rank_grads)
            else:
                rank_grad_norm = 0.0

            # ----------------------------------------------------------
            # Combine losses and back-prop
            # ----------------------------------------------------------
            total_loss_tensor = ce_loss
            if use_ortho:
                total_loss_tensor += ortho_weight * ortho_loss
            if use_rank:
                total_loss_tensor += rank_weight  * rank_loss

            optimizer.zero_grad()
            total_loss_tensor.backward()
            optimizer.step()

            # ----------------------------------------------------------
            # Logging every 200 batches
            # ----------------------------------------------------------
            if batch_idx % 200 == 0:
                with torch.no_grad():
                    _, abs_det = full_rank_regularization(
                        model.encoder, x,
                        epsilon=1e-4, activation_margin=5.0,
                        device=device
                    )
                print(f"  Batch {batch_idx:4d}: "
                      f"CE={ce_loss.item():.4f}  "
                      f"Ortho={ortho_loss.item():.4f}  "
                      f"det={abs_det.item():.4f}\n"
                      f"             Grad-norms  |  "
                      f"CE={ce_grad_norm:.3e}  "
                      f"Ortho={ortho_grad_norm:.3e}  "
                      f"Rank={rank_grad_norm:.3e}")

            # Statistics
            total_loss  += total_loss_tensor.item()
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)
            epoch_ortho += ortho_loss.item()
            epoch_rank  += rank_loss.item()

        avg_loss  = total_loss / len(train_loader)
        train_acc = 100.0 * total_correct / total_samples
        avg_ortho = epoch_ortho / max(1, ortho_count)
        avg_rank  = epoch_rank  / max(1, rank_count)

        print(f"  ➤ epoch avg: loss={avg_loss:.4f}  acc={train_acc:.2f}%  "
              f"ortho={avg_ortho:.4f}  rank={avg_rank:.4f}")

    return model

# ----------------------------------------------------------------------
# 6. Main experiment
# ----------------------------------------------------------------------
if __name__ == "__main__":

    model_both = train_model(use_ortho=True,  use_rank=True, epochs=8)

    print("\nTraining model WITH orthogonality regularisation...")
    model_ortho = train_model(use_ortho=True,  use_rank=False, epochs=8)

    print("\nTraining BASELINE model (no regularisation)...")
    model_base  = train_model(use_ortho=False, use_rank=False, epochs=8)
