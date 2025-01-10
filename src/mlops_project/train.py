import torch
import copy
import os
import torchvision.transforms as transforms
from visualize import plot_performance, save_to_excel
from data import get_dataloaders
from model import ResNet18
import wandb

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



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
          
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    # Save performance plot
    save_path = os.path.join(save_dir, f"{optimizer.__class__.__name__}_performance.png")
    plot_performance(train_losses, val_losses, train_accs, val_accs, optimizer.__class__.__name__, save_path)

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }


def main():
    """
    Entry point for the above training method

    Returns:
        model_performances (dict): Dictionary containing model performances.
    """
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.001,
        "architecture": "ResNet",
        "dataset": "Hotdog/notHodog",
        "epochs": 2,
        }
    )

    # Local data path
    data_path = os.path.normpath("data") 
    transform = transforms.Compose([
        transforms.Resize((128, 128)),            # Resize image to 128x128
        transforms.ToTensor(),                   # Convert to tensor (CxHxW format)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Get DataLoaders
    train_loader, test_loader = get_dataloaders(data_path, batch_size=4, transform=transform)
    # Load the model
    model = ResNet18(num_classes=1)
    # Define the optimizers
    optimizers = torch.optim.Adam(model.parameters(), lr=0.001),
    num_epochs = 2
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
    model_performances = main()
