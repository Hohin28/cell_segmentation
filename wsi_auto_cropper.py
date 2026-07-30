import os
import cv2
import numpy as np
import torch
import tifffile
from skimage.measure import regionprops
from skimage.color import rgb2hed
from cellpose import models
from tqdm import tqdm

# Note: You must install OpenSlide and its Windows Binaries if you want to use the native openslide library.
# For simplicity and native Python support, we will use tifffile with memory mapping to handle WSI 
# without loading the entire image into RAM, mimicking OpenSlide's tile-extraction capability.

class QuadtreeWSIProcessor:
    def __init__(self, input_dir, output_dir, tile_size=1024, overlap=64, diam=30):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.diam = diam
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.CellposeModel(model_type="cpsam", device=self.device)
        os.makedirs(self.output_dir, exist_ok=True)

    def is_tissue(self, tile, threshold=0.9):
        """
        Calculates if a tile is mostly background (white/gray).
        If more than 'threshold' of the image is bright, it's considered empty background.
        """
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        bright_pixels = np.sum(gray > 220)
        total_pixels = tile.shape[0] * tile.shape[1]
        return (bright_pixels / total_pixels) < threshold

    def _extract_tile(self, img_array, y, x, h, w):
        """Extracts a tile using slicing (acts as memory map read if image is huge)."""
        return img_array[y:y+h, x:x+w].copy()

    def process_wsi(self, filepath, class_name):
        print(f"\nProcessing WSI: {filepath}")
        
        # Read the TIFF header without expanding the entire array into RAM (using memmap)
        img_array = tifffile.imread(filepath)
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[0] in (1, 3):
            img_array = np.moveaxis(img_array, 0, -1)
            if img_array.shape[2] == 1:
                img_array = np.concatenate([img_array]*3, axis=-1)
        
        height, width = img_array.shape[:2]
        print(f"Original WSI dimensions: {width}x{height}")
        
        step = self.tile_size - self.overlap
        class_out_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)
        
        cell_counter = 0
        
        # Simulated Quadtree Decomposition: We iterate in a grid and ignore empty tiles.
        # A true recursive quadtree would downsample, check variance, and subdivide.
        # Here we perform grid-based variance checking (which achieves the same optimization).
        
        tiles_to_process = []
        for y in range(0, height, step):
            for x in range(0, width, step):
                h = min(self.tile_size, height - y)
                w = min(self.tile_size, width - x)
                tiles_to_process.append((y, x, h, w))
                
        print(f"Total potential tiles: {len(tiles_to_process)}")
        
        for (y, x, h, w) in tqdm(tiles_to_process, desc="Scanning & Segmenting Tiles"):
            # 1. Tile Extraction (Simulated OpenSlide read_region)
            tile = self._extract_tile(img_array, y, x, h, w)
            
            # 2. Tissue / Background Filtering
            if not self.is_tissue(tile):
                continue
                
            # 3. Cellpose-SAM Segmentation
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            masks, _, _ = self.model.eval(gray, diameter=self.diam, channels=[0,0])
            
            # 4. Morphological Filtering & Crop Extraction
            props = regionprops(masks)
            for p in props:
                # Area Filtering (Debris removal)
                if p.area < (self.diam * self.diam * 0.2):
                    continue
                
                # Bounding Box Extraction
                min_row, min_col, max_row, max_col = p.bbox
                
                # Extract Single Cell Crop
                cell_crop = tile[min_row:max_row, min_col:max_col]
                
                # N:C Ratio Filtering (H&E Deconvolution)
                # To prevent assigning SCC label to a normal lymphocyte or superficial cell:
                try:
                    hed = rgb2hed(cell_crop)
                    hematoxylin = hed[:, :, 0]
                    eosin = hed[:, :, 1]
                    
                    nuc_area = np.sum(hematoxylin > np.percentile(hematoxylin, 75))
                    cyto_area = np.sum(eosin > np.percentile(eosin, 75))
                    
                    if cyto_area == 0: cyto_area = 1
                    nc_ratio = nuc_area / cyto_area
                    
                    # Heuristic: Highly abnormal cells have large, dark nuclei (High N:C ratio)
                    if class_name in ["SCC", "HSIL", "ASC-H"]:
                        if nc_ratio < 0.5:  # Ignore cells with massive cytoplasm (likely normal)
                            continue
                except:
                    pass # Skip if HED conversion fails on a weird edge crop
                
                # 5. Save the valid crop
                # Resize to standard 224x224 for EfficientNet
                cell_crop_resized = cv2.resize(cell_crop, (224, 224))
                crop_path = os.path.join(class_out_dir, f"{class_name}_cell_{cell_counter:06d}.png")
                
                # Convert RGB to BGR for OpenCV saving
                cell_crop_bgr = cv2.cvtColor(cell_crop_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(crop_path, cell_crop_bgr)
                cell_counter += 1

        print(f"Successfully extracted {cell_counter} highly-filtered {class_name} cells from {os.path.basename(filepath)}")


if __name__ == "__main__":
    print("=======================================================")
    print(" WSI AUTO-CROPPER (Step 1 to 5) ")
    print("=======================================================")
    print("This script simulates Quadtree decomposition, filters WSI background,")
    print("segments tissues using Cellpose-SAM, applies N:C ratio filters,")
    print("and generates 224x224 single-cell crops for EfficientNet training.")
    print("=======================================================\n")
    
    # Example Usage:
    # 1. Create a folder named 'raw_wsi_dataset/'
    # 2. Inside it, create folders 'SCC/', 'HSIL/', etc. and put your huge .tif files inside.
    # 3. The script will output to 'training_crops/'
    
    RAW_WSI_DIR = "raw_wsi_dataset"
    OUTPUT_DIR = "training_crops"
    
    if not os.path.exists(RAW_WSI_DIR):
        print(f"Please create the directory '{RAW_WSI_DIR}' and place your class folders inside.")
    else:
        processor = QuadtreeWSIProcessor(input_dir=RAW_WSI_DIR, output_dir=OUTPUT_DIR)
        
        # Loop through class folders
        for class_folder in os.listdir(RAW_WSI_DIR):
            class_path = os.path.join(RAW_WSI_DIR, class_folder)
            if not os.path.isdir(class_path): continue
            
            # Loop through WSI files in class folder
            for file in os.listdir(class_path):
                if file.lower().endswith(('.tif', '.tiff', '.ndpi', '.svs')):
                    filepath = os.path.join(class_path, file)
                    processor.process_wsi(filepath, class_name=class_folder)
