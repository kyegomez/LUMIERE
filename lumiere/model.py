from einops import rearrange, reduce
from torch import Tensor, nn
from zeta.nn.attention import SpatialLinearAttention


class ConvolutionBasedInflationBlock(nn.Module):
    """
    Implements a Convolution-based Inflation Block with fixed BatchNorm layers for normalization.
    Uses einops.rearrange for tensor reshaping for clarity and conciseness.
    Maintains the input tensor's shape, but scales down the time, height, width, and dimensions by the scale_factor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the conv2d and linear layers.
        kernel_size (int or tuple): Size of the conv2d kernel.
        stride (int or tuple): Stride for the conv2d operation.
        padding (int or tuple): Padding for the conv2d operation.
        scale_factor (int): Factor to divide the dimensions by.

    Example:b
        >>> block = ConvolutionBasedInflationBlock(3, 64, (3, 3), 1, 1, scale_factor=2)
        >>> x = torch.randn(1, 2, 224, 224, 3)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 1, 112, 112, 64])  # Note that each dimension is halved if scale_factor is 2
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
        self.out_channels = out_channels

        # Conv2d layer with BatchNorm2d and ReLU activation
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape: (batch, time, height, width, dimensions)
        # Input shape: (batch, time, height, width, dimensions)
        b, t, h, w, d = x.shape

        # Merge the batch and time dimensions and move the channel (dimension) to the correct place for Conv2d
        # The input should be rearranged to (batch*time, dimensions, height, width)
        x = rearrange(x, "b t h w d -> (b t) d h w")

        # Apply 2D convolution, normalization, and activation
        x = self.conv2d(
            x
        )  # Output shape: (batch*time, out_channels, new_height, new_width)
        x = self.norm2d(x)
        x = self.relu(x)

        # Calculate output dimensions and reshape
        _, c, new_h, new_w = x.shape
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        # Reduce dimensions by scale_factor
        x = reduce(
            x,
            (
                "b t c (h h_scale) (w w_scale) -> b (t // t_scale) (h"
                " // h_scale) (w // w_scale) c"
            ),
            "mean",
            h_scale=self.scale_factor,
            w_scale=self.scale_factor,
            t_scale=self.scale_factor,
        )

        # Initialize the linear layer dynamically if it has not been initialized
        if self.linear is None:
            _, t_reduced, h_reduced, w_reduced, _ = x.shape
            flattened_size = (
                t_reduced * h_reduced * w_reduced * self.out_channels
            )
            self.linear = nn.Linear(
                flattened_size, flattened_size // self.scale_factor
            ).to(x.device)

        # Flatten and apply linear layer
        x = rearrange(x, "b t h w c -> b (t h w c)")
        x = self.linear(x)

        # Reshape back to the original dimensions with scale reduction
        x = rearrange(
            x,
            "b (t h w c) -> b t h w c",
            t=t // self.scale_factor,
            h=h // self.scale_factor,
            w=w // self.scale_factor,
            c=self.out_channels,
        )

        return x


class AttentionBasedInflationBlock(nn.Module):
    """
    Attention-based inflation block module.

    Args:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout rate. Defaults to 0.1.

    Attributes:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        attn (SpatialLinearAttention): The spatial linear ablttention module.
        proj (nn.Linear): The linear projection layer.
        norm (nn.LayerNorm): The layer normalization module.

    Example:
        >>> import torch
        >>> from lumiere.model import AttentionBasedInflationBlock
        >>> x = torch.randn(1, 4, 224, 224, 512)
        >>> model = AttentionBasedInflationBlock(dim=512, heads=4, dropout=0.1)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 4, 224, 224, 512])

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout

        # Spatial linear attention for videos of size:
        # batch_size, channels, frames, height, width.
        self.attn = SpatialLinearAttention(
            dim, heads, dim_head=dim // heads, *args, **kwargs
        )

        # Linear projection layer
        self.proj = nn.Linear(dim, dim)

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the AttentionBasedInflationBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        skip = x
        b, t, h, w, d = x.shape

        # Reshape to match the spatial linear attention module
        x = rearrange(x, "b t h w d -> b d t h w")

        # Apply spatial linear attention
        x = self.attn(x)

        # Reshape back to the original shape
        x = rearrange(x, "b d t h w -> b t h w d")

        # Linear projection
        x = nn.Linear(d, d)(x)

        return x + skip
