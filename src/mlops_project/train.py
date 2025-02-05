import torch
import copy
import os
import wandb
import typer
import torchvision.transforms as transforms
from google.cloud import storage
from data import get_dataloaders
from model import ResNet18
from hydra import compose, initialize
from typing import Optional

app = typer.Typer()


@app.command()
def train_model(model, train_loader, test_loader, optimizer, num_epochs):
    """
    Train the model and get the performance results.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        model (torch.nn.Module): Trained model.
        performance (dict): Dictionary containing performance metrics
    """

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    initial_model = copy.deepcopy(model)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Create directory for saving plots
    save_dir = "optimizer_performance_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Training with optimizer: {optimizer.__class__.__name__}")
    model.load_state_dict(initial_model.state_dict())

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("Ready to start training")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # with profile(
            #     activities=[ProfilerActivity.CPU], record_shapes=True
            # ) as prof:  # add ProfilerActivity.CUDA   if we use CUDA

            # prof.export_chrome_trace("trace.json")

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == "train":
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    loss = criterion(outputs, labels.unsqueeze(1).float())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.unsqueeze(1))

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc})
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print(f"Passed epoch {epoch}")

    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)

    print("Read to save model weights")

    # save the best model weights
    torch.save(best_model_wts, "models/best_model.pth")

    """
    Upload the model to GCP cloud storage
    """

    print("Read to upload model to GCP cloud storage")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    bucket_name = "mlops-trained-models"  # GCP bucket name
    source_file_name = os.path.join(project_root, "models/best_model.pth")
    destination_blob_name = "models/model.pth"  # Destination file name in the bucket
    key_file = os.path.join(project_root, "keys/cloud_storage_key.json")  # Path to the service account key file
    upload_to_gcp_bucket(bucket_name, source_file_name, destination_blob_name, key_file)

    performance = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    }

    return model, performance


def upload_to_gcp_bucket(bucket_name, source_file_name, destination_blob_name, key_file):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


@app.command()
def entrypoint(config_name: str = "config.yaml", epochs: Optional[int] = None):
    """
    Entry point for the above training method

    Returns:
        model_performances (dict): Dictionary containing model performances.
    """
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name=config_name)

    if epochs is not None:
        cfg.hyperparameters.epochs = epochs

    try:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    except Exception as e:
        print(f"WandB login failed: {e}")
        return

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.hyperparameters.learning_rate,
            "architecture": "ResNet",
            "dataset": "Hotdog/notHodog",
            "epochs": cfg.hyperparameters.epochs,
        },
    )

    # Local data path
    data_path = os.path.normpath("data")
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize image to 128x128
            transforms.ToTensor(),  # Convert to tensor (CxHxW format)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Get DataLoaders
    train_loader, test_loader = get_dataloaders(data_path, batch_size=4, transform=transform)
    # Load the model
    model = ResNet18(num_classes=1)
    # Define the optimizers
    optimizers = (torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.learning_rate),)
    num_epochs = cfg.hyperparameters.epochs
    model_performances = {}

    for optimizer in optimizers:
        model, performance = train_model(model, train_loader, test_loader, optimizer, num_epochs)
        model_performances[optimizer.__class__.__name__] = {
            "model": model,
            "performance": performance,
        }
    wandb.finish()

    return model_performances


if __name__ == "__main__":
    model_performances = app()
