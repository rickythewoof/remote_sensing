import sys
import torch
import utils
import dataset.dataset as dataset

from rasterio.plot import show

device = utils.test_cuda()
# Load the dataset
train_dataset = dataset.SN6Dataset(root_dir='data/train/AOI_11_Rotterdam', split="train", dtype="PS-RGBNIR")

# Load the data loaders 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)


def train(epoch, model, optimizer, criterion, log_interval = 100):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()}')