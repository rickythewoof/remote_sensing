import sys
import torch
import utils
from model import UNET
import dataset as dataset
import torch.nn as nn
import tqdm

if __name__ == "__main__":
    device = utils.set_cuda_and_seed()

    # Load the dataset

    train_dataset = dataset.SN6Dataset(root_dir='data/train/AOI_11_Rotterdam', split="train", dtype="PS-RGB")
    eval_dataset = dataset.SN6Dataset(root_dir='data/train/AOI_11_Rotterdam', split="val", dtype="PS-RGB")

    # Load the data loaders 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)


    # TODO: Choose the model, loss and the optimizer
    model = UNET(in_channels = 3, out_channels = 1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optimizer.param_groups[0]['lr']*0.9)

def train (train_loader, eval_loader, model, optimizer, criterion, scheduler, device, num_epochs = 10):
    for _ in range(num_epochs):
        train_loss = train_1_epoch(train_loader, model, optimizer, criterion, device)
        print(f"Training Loss: {train_loss}")
        eval_accuracy = evaluate(eval_loader, model, device)
        print(f"Evaluation Accuracy: {eval_accuracy}")
        scheduler.step()
        
def train_1_epoch(data_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count_batches = 0

    bar = tqdm.tqdm(data_loader)
    for data, mask in bar:
        # Move the data to the device
        data, mask = data.to(device), mask.to(device).squeeze(dim=1)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        output = output.squeeze(dim=1)
        # Calculate the loss
        loss = criterion(output, mask)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        total_loss += loss.item()
        count_batches += 1
        bar.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / count_batches  # Returning the average loss for the epoch

def evaluate(data_loader, model, device):
    model.eval()
    num_preds = 0
    true_preds = 0
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            pred = pred.squeeze(dim = 1)
            pred = torch.sigmoid(pred)
            pred_lbl = (pred >= 0.5).long()

            true_preds += torch.sum(pred_lbl == label)
            num_preds += label.shape[0]
    
    accuracy = true_preds/num_preds
    print(f"Accuracy: {100*accuracy:.2f}%")
    return accuracy
