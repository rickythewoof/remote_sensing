import os
import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import torchvision.utils
from shapely.geometry import shape
import tqdm


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
def get_mean_std(path_to_train_data):
    mean = np.zeros(3)
    std = np.zeros(3)
    samples = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    with open(path_to_train_data) as f:
        image_paths = f.read().splitlines()
    
    progress_bar = tqdm.tqdm(total=len(image_paths), desc='Calculating mean and std')
    
    for image_path in image_paths:
        with rasterio.open(image_path) as dataset:
            image = dataset.read()
            image = torch.from_numpy(image).to(device)
            
            for i in range(3):
                mean[i] += torch.mean(image[i].float()).item()
                std[i] += torch.std(image[i].float()).item()
        
        samples += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    mean /= samples
    std /= samples

    return mean, std

def visualize_image(image, mask):
    # Convert image and mask to numpy arrays

    # Transpose the dimensions of image and mask

    # TODO: Visualize the image and mask
    image = np.array(image)
    mask = np.array(mask).squeeze()
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title("Mask")
    plt.show()

def normalize_image(image):
    # Normalize the image
    image = image / 255.0
    return image

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_evals(data_loader, model, device):
    # Get the accuracy and Jaccard index of the model
    true_preds = 0
    num_preds = 0
    dice_score = 0
    jaccard_index = 0
    model.eval()
    
    with torch.no_grad():
        for data, mask in data_loader:
            data = data.to(device)
            mask = mask.to(device).squeeze(dim=1)
            pred = torch.sigmoid(model(data))
            pred = (pred > 0.5).float()
            true_preds += (pred == mask).sum()
            num_preds += pred.numel()
            intersection = (pred * mask).sum()
            union = (pred + mask).sum() - intersection
            
            dice_score += (2*intersection) / (union + intersection + 1e-8)
            jaccard_index += (intersection + 1e-8) / (union + 1e-8)
    
    accuracy = true_preds / num_preds
    dice_score /= len(data_loader)
    jaccard_index /= len(data_loader)
    
    print(f"Accuracy (check): {accuracy:.4f}")
    print(f"Dice Score: {dice_score:.4f}")
    print(f"Jaccard Index: {jaccard_index:.4f}")
    
    model.train()

def save_predictions_as_image(loader, model, device, output_path):
    # Save the predictions as an image in the output_path
    model.eval()
    with torch.no_grad():
        for idx, (data, mask) in enumerate(loader):
            data = data.to(device)
            mask = mask.to(device)
            output = torch.sigmoid(model(data))
            output = (output > 0.5).float()
            torchvision.utils.save_image(output, os.path.join(output_path, f"{idx} - predictions.png"))
            torchvision.utils.save_image(mask, os.path.join(output_path, f"{idx} - masks.png"))
    model.train()
                                
                                         


def save_checkpoint(state, filename="model_checkpoint.pth"):
    print("saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(name, model, optimizer, scheduler):
    print("loading checkpoint")
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])



if __name__ == "__main__":
    print("Calculating the mean and standard deviation of the dataset")
    mean, std = get_mean_std("data/train/AOI_11_Rotterdam/splits/train.txt")
    print("Mean:", mean)
    print("Standard Deviation:", std)