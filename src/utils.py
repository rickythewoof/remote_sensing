import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# Don't start the training process without checking CUDA availability
def test_cuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    if device != torch.device('cuda'):
        sys.exit("CUDA not available.  Exiting.")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    return device

# Get the mean and standard deviation of the dataset for normalization transforms
# TODO: To check
def get_mean_std(root_dir, split, dtype):
    return None


def visualize_image(image, mask):
    # TODO: Visualize the image and mask
    plt.imshow(image)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)