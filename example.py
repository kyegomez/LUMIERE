import torch
from model import ConvolutionBasedInflationBlock


# B, T, H, W, C
x = torch.randn(1, 2, 112, 112, 3)

# Create a ConvolutionBasedInflationBlock
block = ConvolutionBasedInflationBlock(
    in_channels=3,
    out_channels=64,
    kernel_size=(3, 3),
    stride=1,
    padding=1,
    scale_factor=2,
)


# Pass the input tensor through the block
out = block(x)


# Print the output shape
print(out.shape)
