import os
import tqdm
import cv2
from shapely.geometry import shape
import rasterio
import geojson
from PIL import Image, ImageDraw
import numpy as np

def create_masks(image_folder, geojson_folder):
    # Create masks folder if it doesn't exist
    masks_folder = os.path.join(image_folder, '../masks')
    os.makedirs(masks_folder, exist_ok=True)

    # Get list of image files
    image_files = os.listdir(image_folder)

    # Iterate over image files
    for image_file in tqdm.tqdm(image_files, desc='Creating masks'):
        # Get image file path
        image_path = os.path.join(image_folder, image_file)
        image = rasterio.open(image_path)
        # Get corresponding geojson file path
        geojson_path = image_path.strip().replace("/PS-RGB/", "/geojson_buildings/").replace("PS-RGB","Buildings").replace(".tif", ".geojson")

        # Create mask using geojson data
        mask = geojson_to_mask(image, geojson_path)

        # Save mask to masks folder
        mask_file = image_file.replace('.tif', '.png')
        mask_path = os.path.join(masks_folder, mask_file)
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)


def geojson_to_mask(image, geojson_path):

    # Leggi il file GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = geojson.load(f)

    # Crea un'immagine binaria vuota
    mask = Image.new('L', (image.shape[0], image.shape[1]), 0)
    draw = ImageDraw.Draw(mask)

    # Disegna i poligoni sulla maschera
    for feature in geojson_data['features']:
        geom = shape(feature['geometry'])
        if geom.geom_type == 'Polygon':
            coords = get_coords(image.transform, geom.exterior.coords)
            draw.polygon(coords, outline=1024, fill=255)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = get_coords(image.transform, poly.exterior.coords)
                draw.polygon(coords, outline=1024, fill=255)

    # Converti l'immagine PIL in un array NumPy
    mask = np.array(mask)
    return mask

def get_coords(transform, coords):
    new_coords = []
    for coord in coords:
        print(coord)
        x, y = coord[:2]
        px, py = ~transform * (x, y)
        new_coords.append((int(px), int(py)))
    return new_coords

if __name__ == "__main__":
    image_folder = 'data/train/AOI_11_Rotterdam/PS-RGB'
    geojson_folder = 'data/train/AOI_11_Rotterdam/geojson_buildings'
    create_masks(image_folder, geojson_folder)
    print("Masks created successfully!")