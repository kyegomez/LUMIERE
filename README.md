[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Lumiere 
Implementation of the text to video model LUMIERE from the paper: "A Space-Time Diffusion Model for Video Generation" by Google Research. I will mostly be implementing the modules from the diagram a and b in figure 4

## Install
`pip install lumiere`


## Usage
```python
import torch
from lumiere.model import AttentionBasedInflationBlock

# B, T, H, W, D
x = torch.randn(1, 4, 224, 224, 512)

# Model
model = AttentionBasedInflationBlock(dim=512, heads=4, dropout=0.1)

# Forward pass
out = model(x)

# print
print(out.shape)  # Expected shape: [1, 4, 224, 224, 3]

```


# License
MIT
