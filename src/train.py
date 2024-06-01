import torch
import dataset as dataset
from tqdm import tqdm
        
def train(train_loader, model, optimizer, criterion, scaler, scheduler, device):
    model.train()

    bar = tqdm(train_loader)
    total_loss = 0
    num_batches = 0
    for data, mask in bar:
        # Move the data to the device
        data = data.to(device)
        mask = mask.to(device).squeeze(dim = 1)
        # Scale the data
        # Zero the gradients
        # Forward pass
        with torch.cuda.amp.autocast():
            pred = model(data).squeeze(dim = 1)
            # Calculate the loss
            loss = criterion(pred, mask)
            if loss != loss:
                raise ValueError("Loss is NaN, something is VERY wrong, stopping training")
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Update the weights
        scaler.step(optimizer)
        if(scheduler is not None):
            scheduler.step()
        scaler.update()
        # Accumulate the loss
        total_loss += loss.item()
        num_batches += 1
        # Update the progress bar
        bar.set_description(f"Loss: {loss.item():.4f}")
    
    # Calculate the average loss
    average_loss = total_loss / num_batches
    return average_loss