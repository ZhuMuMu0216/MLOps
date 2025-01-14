import os
import pytest
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from src.mlops_project.data import HotdogNotHotdog, get_dataloaders

# Define global test parameters
PARENT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)  # Return the parent directory of the current file
DATA_PATH = os.path.join(PARENT_DIR, "data")  # Make it work on Unix and Windows
BATCH_SIZE = 4
TRANSFORM = Compose([Resize((128, 128)), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def test_directory_structure():
    """Test if the directory structure is as expected."""
    train_path = os.path.join(DATA_PATH, "train")
    test_path = os.path.join(DATA_PATH, "test")

    assert os.path.exists(train_path), "Training directory does not exist."
    assert os.path.exists(test_path), "Testing directory does not exist."
    assert len(os.listdir(train_path)) > 0, "Training directory is empty."
    assert len(os.listdir(test_path)) > 0, "Testing directory is empty."
    print("Directory structure is correct.")


@pytest.fixture
def dataset_train():
    """Fixture for the training dataset."""
    return HotdogNotHotdog(data_path=DATA_PATH, train=True, transform=TRANSFORM)


@pytest.fixture
def dataset_test():
    """Fixture for the testing dataset."""
    return HotdogNotHotdog(data_path=DATA_PATH, train=False, transform=TRANSFORM)


@pytest.fixture
def dataloaders():
    """Fixture for DataLoaders."""
    return get_dataloaders(data_path=DATA_PATH, batch_size=BATCH_SIZE, transform=TRANSFORM)


def test_dataset_initialization(dataset_train, dataset_test):
    """Test if the dataset initializes properly and counts samples correctly."""
    assert len(dataset_train) > 0, "Training dataset should not be empty."
    assert len(dataset_test) > 0, "Testing dataset should not be empty."
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Testing dataset size: {len(dataset_test)}")


def test_dataloader_batch_shape(dataloaders):
    """Test if the DataLoader returns batches with the correct shape."""
    train_loader, test_loader = dataloaders

    # Test training DataLoader
    for images, labels in train_loader:
        assert images.shape == (BATCH_SIZE, 3, 128, 128), "Image batch shape is incorrect."
        assert len(labels) == BATCH_SIZE, "Label batch size is incorrect."
        print(f"Training batch - Image shape: {images.shape}, Labels: {labels}")
        break

    # Test testing DataLoader
    for images, labels in test_loader:
        assert images.shape == (BATCH_SIZE, 3, 128, 128), "Image batch shape is incorrect."
        assert len(labels) == BATCH_SIZE, "Label batch size is incorrect."
        print(f"Testing batch - Image shape: {images.shape}, Labels: {labels}")
        break
