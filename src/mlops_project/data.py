import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image


class HotdogNotHotdog(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        """
        Initialize the dataset
        Args:
            data_path (str): Root directory where the dataset is stored
            train (bool): If True, load training data; otherwise, load testing data
            transform (callable): Data augmentation or preprocessing function
        """
        self.data_path = os.path.join(data_path, "train" if train else "test")
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Get the names of all subfolders (e.g., hotdog, not_hotdog) as class names
        self.class_to_label = {cls_name: idx for idx, cls_name in enumerate(os.listdir(self.data_path))}
        # Traverse each subfolder to retrieve file paths and corresponding labels
        for class_name, label in self.class_to_label.items():
            class_dir = os.path.join(self.data_path, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(".jpg"):  # Process only .jpg files
                    img_path = os.path.join(class_dir, img_file)
                    image = Image.open(img_path).convert("RGB")  # Convert to RGB mode
                    if self.transform:
                        image = self.transform(image)
                    self.image_paths.append(image)
                    self.labels.append(label)

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Return a sample based on the index
        Args:
            idx (int): Sample index
        Returns:
            image (torch.Tensor): Image data (CxHxW format)
            label (int): Corresponding label for the image
        """
        # Get the preprocessed image
        image = self.image_paths[idx]

        # Get the corresponding label
        label = self.labels[idx]
        return image, label


def get_dataloaders(data_path, batch_size=4, transform=None):
    """
    Return DataLoaders for training and testing datasets.
    Args:
        data_path (str): Root directory where the dataset is stored.
        batch_size (int): Number of samples per batch.
        transform (callable): Data augmentation or preprocessing function.
    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
    """
    # Create training and testing datasets
    train_dataset = HotdogNotHotdog(data_path=data_path, train=True, transform=transform)
    test_dataset = HotdogNotHotdog(data_path=data_path, train=False, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == "__main__":
    # Local data path
    # Normalize data_path for cross-platform compatibility
    data_path = os.path.normpath("data")  # Automatically adjusts path for current OS

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize image to 128x128
            transforms.ToTensor(),  # Convert to tensor (CxHxW format)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Get DataLoaders
    train_loader, test_loader = get_dataloaders(data_path, batch_size=4, transform=transform)

    # Print the size of the datasets
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Test a single batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Image shape: {images.shape}")  # Should be [batch_size, 3, 128, 128]
        print(f"  Labels: {labels}")
        break
