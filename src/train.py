import sys
import torch
import utils
from model import UNET
import dataset as dataset
import torch.nn as nn
import tqdm.notebook as tqdm

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

def train (train_loader, eval_loader, model, optimizer, criterion, num_epochs = 10):
    for epoch in range(num_epochs):
        loss = train_1_epoch(train_loader, model, optimizer, criterion)
        print(f"Epoch: {epoch} Loss: {loss}")
        accuracy = evaluate(eval_loader, model)
        utils.save_model(model, optimizer, epoch, loss, accuracy)
        scheduler.step()
        if (False): # Early stopping condition!
            utils.save_model(model, optimizer, epoch, loss, accuracy)

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

def evaluate(data_loader, model):
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
