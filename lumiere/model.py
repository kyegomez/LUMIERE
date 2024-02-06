from einops import rearrange
from torch import nn, Tensor
from zeta.nn.attention import SpatialLinearAttention

from einops import rearrange
from torch import nn, Tensor


class ConvolutionBasedInflationBlock(nn.Module):
    """
    Implements a Convolution-based Inflation Block with fixed BatchNorm layers for normalization.
    Uses einops.rearrange for tensor reshaping for clarity and conciseness.
    Maintains the input tensor's dimensions, but scales down the temporal (T),
    height (H), and width (W) dimensions by a scale factor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the conv2d and linear layers.
        kernel_size (int or tuple): Size of the conv2d kernel.
        stride (int or tuple): Stride for the conv2d operation.
        padding (int or tuple): Padding for the conv2d operation.
        scale_factor (int): Factor to scale down the T, H, and W dimensions by.

    Example:
        >>> block = ConvolutionBasedInflationBlock(3, 64, (3, 3), 1, 1, scale_factor=2)
        >>> x = torch.randn(1, 2, 224, 224, 3)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 1, 112, 112, 64])
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

        # Conv2d layer with BatchNorm2d and ReLU activation
        self.conv2d = nn.Conv2d(
            in_channels,  # Number of input channels
            out_channels,  # Number of output channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Linear projection layer
        # The number of features before the linear layer needs to be calculated dynamically based on the input size
        self.linear = None

    def forward(self, x: Tensor):
        # Input shape: (batch, time, height, width, dimensions)
        b, t, h, w, d = x.shape

        # Reshape input to combine the dimensions and time into the batch dimension for 2D convolution
        x = rearrange(x, "b t h w d -> (b t) h w d")

        # Apply 2D convolution, normalization, and activation
        x = self.conv2d(
            x
        )  # Shape after conv: (batch*time, out_channels, new_height, new_width)
        x = self.norm2d(x)
        x = self.relu(x)

        # Calculate the new shape after the convolution
        _, _, new_h, new_w = x.shape
        new_t = t // self.scale_factor
        new_h = new_h // self.scale_factor
        new_w = new_w // self.scale_factor

        # Check if linear layer has been defined, define it if not
        if self.linear is None:
            flattened_size = new_h * new_w * self.out_channels
            self.linear = nn.Linear(
                flattened_size, flattened_size // self.scale_factor
            ).to(x.device)

        # Reshape x to collapse the output H and W dimensions and uncollapse the batch and time dimensions
        x = rearrange(
            x,
            "(b t) c h w -> b (t h w c)",
            b=b,
            t=new_t,
            h=new_h,
            w=new_w,
        )

        # Apply the linear projection
        x = self.linear(x)

        # Reshape x to the original dimensions with scaled T, H, and W
        x = rearrange(
            x,
            "b (t h w c) -> b t h w c",
            t=new_t,
            h=new_h,
            w=new_w,
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
        attn (SpatialLinearAttention): The spatial linear attention module.
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
            dim,
            heads,
            dim_head=dim // heads,
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
