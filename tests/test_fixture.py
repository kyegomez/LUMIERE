import torch
from lumiere.model import AttentionBasedInflationBlock

import pytest

@pytest.fixture
def lumiere_fixture():
    # B, T, H, W, D
    x = torch.randn(1, 4, 224, 224, 512)
    # Model
    model = AttentionBasedInflationBlock(dim=512, heads=4, dropout=0.1)
    # Forward pass
    return model(x)

def test_with_fixture(lumiere_fixture):
    assert list(lumiere_fixture.shape) == [1, 4, 224, 224, 512]
