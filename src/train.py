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

        
def train(train_loader, model, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    count_batches = 0

    bar = tqdm.tqdm(train_loader)
    for data, mask in bar:
        # Move the data to the device
        data, mask = data.to(device), mask.to(device).squeeze(dim=1)
        # Scale the data
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        with torch.cuda.amp.autocast():
            output = model(data)
            output = output.squeeze(dim=1)
            # Calculate the loss
            loss = criterion(output, mask)
        # Backward pass
        scaler.scale(loss).backward()
        # Update the weights
        scaler.step(optimizer)
        scaler.update()
        bar.set_description(f"Loss: {loss.item():.4f}")