# full_rank_test_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functorch import jacrev

# ----------------------------------------------------------------------
# 1.  Hyper-parameters
# ----------------------------------------------------------------------
in_dim       = 28 * 28        # MNIST flattened
hidden_dims  = [256, 256]     # any list or int
out_dim      = 10             # digits 0-9
batch_size   = 128            # for both train & Jacobian test
target_acc   = 99.0           # target training accuracy (%)
lr           = 1e-3
device       = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# 2.  Model: import the full_rank_mlp defined earlier
#     (place your implementation in full_rank_mlp.py or inline)
# ----------------------------------------------------------------------
from common.full_rank_layers import full_rank_mlp
model = full_rank_mlp(in_dim, hidden_dims, out_dim).to(device)

# ----------------------------------------------------------------------
# 3.  Data
# ----------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.view(-1))])  # flatten
train_set = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

# ----------------------------------------------------------------------
# 4.  Function to test Jacobian rank on multiple batches
# ----------------------------------------------------------------------
def test_jacobian_rank(model, test_loader, num_batches=10):
    """Test Jacobian rank on specified number of batches"""
    model.eval()
    
    def f_single(x):
        return model(x)
    
    jacobian_fn = jacrev(f_single)
    all_ranks = []
    
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            x_batch = x_batch.to(device)
            # Compute batched Jacobians: shape (B, out_dim, in_dim)
            jacobians = torch.vmap(jacobian_fn)(x_batch)
            # Rank per sample
            ranks = torch.linalg.matrix_rank(jacobians)
            all_ranks.append(ranks)
    
    # Combine all ranks
    all_ranks = torch.cat(all_ranks)
    rank_min = all_ranks.min().item()
    rank_mean = all_ranks.float().mean().item()
    
    print(f"  Jacobian ranks on {len(all_ranks)} samples from {num_batches} batches:")
    print(f"    min rank  = {rank_min}")
    print(f"    mean rank = {rank_mean:.2f}")
    print(f"    target output dim = {out_dim}")
    
    if rank_min == out_dim:
        print("  ✅  PASS: network stayed full-rank on these batches.")
        return True
    else:
        print("  ❌  FAIL: network dropped rank.")
        return False

# ----------------------------------------------------------------------
# 5.  Optimiser & training loop with accuracy tracking
# ----------------------------------------------------------------------
optim = torch.optim.Adam(model.parameters(), lr=lr)

epoch = 0
train_acc = 0.0

while train_acc < target_acc:
    epoch += 1
    model.train()
    
    total_correct = 0
    total_samples = 0
    epoch_loss = 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += y.size(0)
        epoch_loss += loss.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    # Calculate training accuracy
    train_acc = (total_correct / total_samples) * 100
    avg_loss = epoch_loss / len(train_loader)
    
    print(f"\nEpoch {epoch} – train-loss {avg_loss:.4f} – train-acc {train_acc:.2f}%")
    
    # Test Jacobian rank after every epoch on 10 batches
    is_full_rank = test_jacobian_rank(model, test_loader, num_batches=10)
    
    if train_acc >= target_acc:
        print(f"\n🎯 Target accuracy of {target_acc}% reached!")
        break

print(f"\nTraining completed after {epoch} epochs with {train_acc:.2f}% accuracy.")

