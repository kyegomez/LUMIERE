import torch
from lumiere.model import ConvolutionBasedInflationBlock

# Example usage:
# scale_factor must be a divisor of T, H, and W for the example to work correctly
block = ConvolutionBasedInflationBlock(
    3, 64, (3, 3), (2, 2), (1, 1), scale_factor=2
)
x = torch.randn(1, 4, 224, 224, 3)
out = block(x)
print(out.shape)  # Expected shape: [1, 2, 112, 112, 64]
