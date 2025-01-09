import pytest
import torch
from src.mlops_project.model import ResNet18


@pytest.mark.parametrize("batch_size", [32, 64])
def test_resnet18_output_shape(batch_size: int):
    model = ResNet18(num_classes=1)

    # Create random input data with shape (batch_size, channels, height, width)
    input_data = torch.randn(batch_size, 3, 128, 128)

    # Get the model output
    output = model(input_data)

    # Check the output shape
    assert output.shape == (batch_size, 1), f"Expected output shape ({batch_size}, 1), but got {output.shape}"
