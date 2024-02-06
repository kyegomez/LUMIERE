import torch
from lumiere.model import ConvolutionBasedInflationBlock

# B, T, H, W, D
x = torch.randn(1, 4, 224, 224, 512)

# Create the model
model = ConvolutionBasedInflationBlock(
    in_channels=512,
    out_channels=512,
    kernel_size=3,
    stride=1,
    padding=1,
    scale_factor=2,
)


# Forward pass
out = model(x)

# Print the output shape
print(out.shape)
