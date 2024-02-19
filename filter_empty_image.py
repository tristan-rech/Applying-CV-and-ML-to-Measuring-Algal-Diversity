import cv2
import os
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt

def process_images(directory_path):
    # Define the output directories
    cell_dir = Path(directory_path) / 'cell'
    empty_dir = Path(directory_path) / 'empty'
    
    # Create output directories if they don't exist
    cell_dir.mkdir(exist_ok=True)
    empty_dir.mkdir(exist_ok=True)
    
    # List all PNG images in the given directory
    png_images = list(Path(directory_path).glob('*.png'))
    
    for image_path in png_images:
        # Load the image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Determine if edges are present
        # Here we use a simple thresholding approach to categorize
        # You might want to adjust the threshold based on your needs
        edge_presence = np.mean(edges) > 0  # Thresholding condition
        
        # Move images to respective directories based on edge presence
        if edge_presence:
            shutil.move(str(image_path), cell_dir / image_path.name)
        else:
            shutil.move(str(image_path), empty_dir / image_path.name)

def process_images_sobel(directory_path):
    # Define the output directories
    cell_dir = Path(directory_path) / 'cell'
    empty_dir = Path(directory_path) / 'empty'
    
    # Create output directories if they don't exist
    cell_dir.mkdir(exist_ok=True)
    empty_dir.mkdir(exist_ok=True)
    
    # List all PNG images in the given directory
    png_images = list(Path(directory_path).glob('*.png'))
    
    for image_path in png_images:
        # Load the image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection using Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Edge Detection on the Y axis
        
        # Combine the Sobel X and Y results
        sobel_combined = cv2.magnitude(sobelx, sobely)
        
        # Determine if edges are present
        # Adjust the threshold based on your needs
        edge_presence = np.mean(sobel_combined) > 0  # Thresholding condition
        
        # Move images to respective directories based on edge presence
        if edge_presence:
            shutil.move(str(image_path), cell_dir / image_path.name)
        else:
            shutil.move(str(image_path), empty_dir / image_path.name)
        
        # display
        plt.figure(figsize=(12, 8))
        plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
        plt.subplot(235), plt.imshow(edge_presence, cmap='gray'), plt.title('Edge Detection')
        plt.show()

# Example usage
directory_path = 'data\ET22-05\ET-F\ET-F1 Final\split'
#process_images(directory_path)
process_images_sobel(directory_path)
