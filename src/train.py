import torch
import dataset as dataset
from tqdm import tqdm
        
def train(train_loader, model, optimizer, criterion, scaler, scheduler, device):
    model.train()

    bar = tqdm(train_loader)
    for data, mask in bar:
        # Move the data to the device
        data = data.to(device)
        mask = mask.to(device).squeeze(dim = 1)
        # Scale the data
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        with torch.cuda.amp.autocast():
            output = model(data)
            output = output.squeeze(dim = 1)
            # Calculate the loss
            loss = criterion(output, mask)
        # Backward pass
        scaler.scale(loss).backward()
        # Update the weights
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        bar.set_description(f"Loss: {loss.item():.4f}")