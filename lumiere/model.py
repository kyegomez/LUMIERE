from einops import rearrange, reduce
from torch import nn, Tensor
from zeta.nn import MultiQueryAttention

from einops import rearrange, reduce
from torch import nn, Tensor


class ConvolutionBasedInflationBlock(nn.Module):
    """
    Implements a Convolution-based Inflation Block with fixed BatchNorm layers for normalization.
    Uses einops.rearrange and reduce for tensor reshaping for clarity and conciseness.
    Maintains the input tensor's shape, but scales the time, height, width, and dimensions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the conv2d and linear layers.
        kernel_size (int or tuple): Size of the conv2d kernel.
        stride (int or tuple): Stride for the conv2d operation.
        padding (int or tuple): Padding for the conv2d operation.
        scale_factor (int): Factor to scale the dimensions by.

    Example:
        >>> block = ConvolutionBasedInflationBlock(3, 64, (3, 3), 1, 1, scale_factor=2)
        >>> x = torch.randn(1, 2, 224, 224, 3)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 2, 112, 112, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale_factor,
    ):
        super(ConvolutionBasedInflationBlock, self).__init__()
        self.scale_factor = scale_factor

        # Assume kernel_size, stride, padding are tuples (for height and width)
        # Conv2d layer with BatchNorm2d and ReLU activation
        self.conv2d = nn.Conv2d(
            in_channels
            * scale_factor,  # Scale the in_channels by the scale factor
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Since we are scaling the input dimensions, we need to account for this in the number of output features
        self.num_features_before_linear = (
            out_channels * scale_factor * scale_factor
        )

        # Linear projection layer
        self.linear = nn.Linear(
            self.num_features_before_linear,
            out_channels * scale_factor,
        )

    def forward(self, x: Tensor):
        # Input shape: (batch, time, height, width, dimensions)
        # The input x is expected to be a 5D Tensor

        # Combine batch and time dimensions and scale height and width
        x = rearrange(
            x,
            "b t h w d -> (b t) (h w d)",
            h=self.scale_factor,
            w=self.scale_factor,
        )

        # Apply 2D convolution, normalization, and activation
        x = self.conv2d(x)
        x = self.norm2d(x)
        x = self.relu(x)

        # Reshape back to original batch and time dimensions and reduce height and width according to scale factor
        x = rearrange(
            x,
            "(b t) c h w -> b t (h h_scale) (w w_scale) c",
            b=x.size(0) // self.scale_factor,
            t=self.scale_factor,
            h_scale=1 / self.scale_factor,
            w_scale=1 / self.scale_factor,
        )

        # Flatten the output for linear projection
        x = reduce(x, "b t h w c -> b t (h w c)", "sum")

        # Apply linear projection to scale dimensions back to original
        x = self.linear(x)

        # Reshape to match the input dimensions: (batch, time, height, width, dimensions)
        x = rearrange(
            x,
            "b t (h w d) -> b t h w d",
            h=x.size(2) // (self.scale_factor * self.out_channels),
            w=self.scale_factor,
            d=self.out_channels,
        )

        return x


class AttentionBasedInflationBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dropout = dropout

        # MultiQueryAttention layer
        self.attn = MultiQueryAttention(
            channels, heads, *args, **kwargs
        )

        # Linear projection layer
        self.proj = nn.Linear(channels, channels)

        # Norm
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor):
        skip = x
        b, c, h, w = x.shape

        # Reshape the input tensor for the attention layer
        x = reduce(x, "b c h w -> b (h w) c", "mean")
        x = print(f"X: {x.shape}")

        # Attention
        x, _, _ = self.attn(x, x, x)
        print(f"X: {x.shape}")
        x = self.norm(x)

        # Linear projection
        x = self.proj(x)
        x = self.norm(x)

        # Reshape the output tensor
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x + skip
