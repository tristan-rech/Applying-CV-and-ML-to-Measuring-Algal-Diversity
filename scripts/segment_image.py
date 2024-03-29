import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import glob
import shutil
from pathlib import Path
import seaborn as sns
import numpy as np
import math

def segment_image(input_dir, output_dir, min_area=100):
    """
    Processes .tif files (except binary) in the input directory, extracting cell images from each
    and saving them in sorted order in the output directory.

    Currently takes ONE directory and outputs to ONE directory

    Hardcoded the label (WK1-..., WK2-..., IN-...)

    Parameters:
    - input_dir: The directory containing the .tif files to process.
    - output_dir: The directory where the rectangle images will be saved.
    - min_area: The minimum area threshold for a contour to be considered a valid rectangle.
    
    Returns:
    - extracted_counts: A dictionary with file numbers as keys and the count of extracted images as values.
    """
    # ensure the output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # initialize a counter for the filenames
    file_counter = 1
    # initialize a dictionary for the counts
    extracted_counts = {}

    # get all .tif files in the input directory
    tif_files = sorted(glob.glob(os.path.join(input_dir, '*.tif')))

    for tif_file in tif_files:
        # filter binary images
        if "bin" in tif_file:
            continue

        # read the image
        image = cv2.imread(tif_file)

        # extract the file number from the filename
        file_number = tif_file.split('_')[-1].split('.')[0]

        print(f"Processing {tif_file}...")
        
        # convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply a binary threshold to get cell image areas
        _, binary_threshold = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # find contours of the cell image areas
        contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filter out contours that are too small
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # sort contours based on the position of their bounding box's top-left corner
        sorted_contours = sorted(valid_contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

        images_from_current_tif = 0
        # extract and save the rectangles
        for i, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            sorted_extracted_image = image[y:y+h, x:x+w]
            file_path = os.path.join(output_dir, f'ET-F1 Final_{str(file_counter).zfill(6)}.png')
            cv2.imwrite(file_path, sorted_extracted_image)
            file_counter += 1
            images_from_current_tif += 1
        
        # save the count for the current file
        extracted_counts[file_number] = images_from_current_tif

        print(f"Extracted {images_from_current_tif} images from {tif_file}.")
        # if tif_file == tif_files[0]:
        #     plt.figure(figsize=(15, 10))

        #     plt.subplot(2, 2, 1)
        #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     plt.title('1. Original Image')

        #     plt.subplot(2, 2, 2)
        #     plt.imshow(gray_image, cmap='gray')
        #     plt.title('2. Grayscale Conversion')

        #     plt.subplot(2, 2, 3)
        #     plt.imshow(binary_threshold, cmap='gray')
        #     plt.title('3. Binary Threshold')

        #     # Drawing contours on a copy of the original image
        #     contour_image = image.copy()
        #     cv2.drawContours(contour_image, sorted_contours, -1, (0, 255, 0), 3)

        #     plt.subplot(2, 2, 4)
        #     plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        #     plt.title('4. Contours Highlighted')

        #     plt.tight_layout()
        #     plt.savefig('image_processing_steps.png')
        #     plt.show()
    return extracted_counts

i#nput_directory = 'data\ET22-05\ET-F\ET-F1 Final' # hard coded to run through one folder of the images
#output_directory = 'data\ET22-05\ET-F\ET-F1 Final\split' # output to new folder
#extracted_counts = segment_image(input_directory, output_directory)
