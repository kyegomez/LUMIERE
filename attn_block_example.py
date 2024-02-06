import torch 
from lumiere.model import AttentionBasedInflationBlock

# B, T, H, W, D
x = torch.randn(1, 4, 224, 224, 512)

# Model
model = AttentionBasedInflationBlock(
    dim=512, heads=4, dropout=0.1
)

# Forward pass
out = model(x)

# print
print(out.shape)  # Expected shape: [1, 4, 224, 224, 3]