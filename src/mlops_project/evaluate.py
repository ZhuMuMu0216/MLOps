import torch

def evaluate_model(model, test_loader, device):
    """
    evaluate_model function to evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run the evaluation on.
    
    Returns:
        epoch_loss (float): Loss value for the test dataset.
        epoch_acc (float): Accuracy value for the test dataset
    """

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            loss = criterion(outputs, labels.unsqueeze(1).float())
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.unsqueeze(1))
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    
    return epoch_loss, epoch_acc
