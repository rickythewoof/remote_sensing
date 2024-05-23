import sys
import torch
import utils
import dataset.dataset as dataset
import torch.nn as nn
import tqdm

from rasterio.plot import show

device = utils.test_cuda()

# Load the dataset

train_dataset = dataset.SN6Dataset(root_dir='data/train/AOI_11_Rotterdam', split="train", dtype="PS-RGBNIR")

# Load the data loaders 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)


# TODO: Choose the model, loss and the optimizer
model = None

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train (data_loader, model, optimizer, criterion, num_epochs = 10):
    for epoch in range(num_epochs):
        loss = train_1_epoch(data_loader, model, optimizer, criterion)
        print(f"Epoch: {epoch} Loss: {loss}")

def train_1_epoch(data_loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    count_batches = 0
    bar = tqdm(count_batches, total=len(data_loader))
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(dim = 1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()
        count_batches+=1
        bar.set_description(f"Loss: {loss.item()}")
        bar.update()
    return total_loss/count_batches # Returning the average loss for the epoch

def evaluate(data_loader, model, criterion):
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

def save_model(model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, f"model_checkpoint_{epoch}.pt")

def load_model(name, model, optimizer):
    model = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss