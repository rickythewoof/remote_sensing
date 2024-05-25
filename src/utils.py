import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import json
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
def create_segmentation_mask(image, geojson_path):
    with open(geojson_path) as f:
        geojson_data = json.load(f)
    mask = np.zeros_like(image[:, :, 0])

    # Iterate over the features in the geojson file
    for feature in geojson_data['features']:
        # Convert the feature geometry to a shapely object
        geometry = shape(feature['geometry'])
        if geometry.geom_type == 'Polygon':    
            mask = rasterio.features.geometry_mask([geometry], out_shape=mask.shape, transform=rasterio.Affine(1, 0, 0, 0, -1, 0), invert=True)
    return mask

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
    mask = np.array(mask)
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
    print("Calculating the mean and standard deviation of the dataset")
    mean, std = get_mean_std("data/train/AOI_11_Rotterdam/splits/train.txt")
    print("Mean:", mean)
    print("Standard Deviation:", std)