import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from dataset import SN6Dataset

# Don't start the training process without checking CUDA availability
def set_cuda_and_seed():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    if device != torch.device('cuda'):
        sys.exit("CUDA not available.  Exiting.")
    set_seed(42) # Set the seed to 42, which is the answer to everything
    return device

# Get the mean and standard deviation of the dataset for normalization transforms
# TODO: To check
def get_mean_std(root_dir, split):
    pass
    return None


def visualize_image(image, mask):
    # TODO: Visualize the image and mask
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")
    plt.show()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, optimizer, epoch, loss, f1_score):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'f1_score': f1_score
    }
    torch.save(checkpoint, f"model_checkpoint_{epoch}.pt")

def load_model(name, model, optimizer):
    model = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

if __name__ == "__main__":
    dataset = SN6Dataset(root_dir='data/train/AOI_11_Rotterdam', split="train", dtype="PS-RGB")
    image, mask = dataset[0]
    visualize_image(image)