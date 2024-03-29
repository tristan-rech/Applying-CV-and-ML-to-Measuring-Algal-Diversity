import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def has_significant_edge_near_border(image, border_size=5, edge_threshold=50):
    # Apply edge detection
    edges = cv2.Canny(image, 100, 200)

    # Define border regions (top, bottom, left, right)
    top_border = edges[:border_size, :]
    bottom_border = edges[-border_size:, :]
    left_border = edges[:, :border_size]
    right_border = edges[:, -border_size:]

    # Calculate the sum of edges in the border regions
    border_edge_sum = np.sum(top_border) + np.sum(bottom_border) + np.sum(left_border) + np.sum(right_border)

    # Determine if there is a significant edge near the border
    if border_edge_sum > edge_threshold:
        return True  # Significant edge detected near border
    else:
        return False  # No significant edge near border

def find_partial_cell_images(directory_path):
    partial_cell_images = []
    for image_name in os.listdir(directory_path):
        if image_name.endswith('.png'):
            image_path = os.path.join(directory_path, image_name)
            image = preprocess_image(image_path)
            if has_significant_edge_near_border(image):
                partial_cell_images.append(image_name)
    return partial_cell_images

def visualize_edge_detection(image_path, border_size=5, edge_threshold=50):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(image_path)
    
    # Apply edge detection
    edges = cv2.Canny(image, 100, 200)

    # Calculate the edge sum in border regions
    top_border_sum = np.sum(edges[:border_size, :])
    bottom_border_sum = np.sum(edges[-border_size:, :])
    left_border_sum = np.sum(edges[:, :border_size])
    right_border_sum = np.sum(edges[:, -border_size:])
    
    # Highlight border regions on the original image
    cv2.rectangle(original_image, (0, 0), (original_image.shape[1], border_size), (0, 255, 0), 2)  # Top border
    cv2.rectangle(original_image, (0, original_image.shape[0] - border_size), (original_image.shape[1], original_image.shape[0]), (0, 255, 0), 2)  # Bottom border
    cv2.rectangle(original_image, (0, 0), (border_size, original_image.shape[0]), (0, 255, 0), 2)  # Left border
    cv2.rectangle(original_image, (original_image.shape[1] - border_size, 0), (original_image.shape[1], original_image.shape[0]), (0, 255, 0), 2)  # Right border

    # Show original image with highlighted borders
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Highlighted Borders')

    # Show detected edges
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Detected Edges')
    plt.show()

    # Print the sum of edges in the border regions
    # print(f"Top border edge sum: {top_border_sum}")
    # print(f"Bottom border edge sum: {bottom_border_sum}")
    # print(f"Left border edge sum: {left_border_sum}")
    # print(f"Right border edge sum: {right_border_sum}")

    # Check if any border region exceeds the threshold
    if any(border_sum > edge_threshold for border_sum in [top_border_sum, bottom_border_sum, left_border_sum, right_border_sum]):
        print("Significant edge detected near border.")
    else:
        print("No significant edge near border.")

# Adjust the path as needed
directory_path = 'data/ET22-05/ET-F/ET-F1 Final/split/partial'
partial_cell_images = find_partial_cell_images(directory_path)
print("Partial cell images:", partial_cell_images)

# image_path = 'data\ET22-05\ET-F\ET-F1 Final\split\partial\ET-F1 Final_000087.png'
# visualize_edge_detection(image_path, border_size=5, edge_threshold=50)

# image_path = 'data\ET22-05\ET-F\ET-F1 Final\split\partial\ET-F1 Final_000366.png'
# visualize_edge_detection(image_path, border_size=5, edge_threshold=50)

# image_path = 'data\ET22-05\ET-F\ET-F1 Final\split\partial\ET-F1 Final_003128.png'
# visualize_edge_detection(image_path, border_size=5, edge_threshold=50)

# image_path = 'data\ET22-05\ET-F\ET-F1 Final\split\partial\ET-F1 Final_003163.png'
# visualize_edge_detection(image_path, border_size=5, edge_threshold=50)

# image_path = 'data\ET22-05\ET-F\ET-F1 Final\split\partial\ET-F1 Final_003436.png'
# visualize_edge_detection(image_path, border_size=5, edge_threshold=50)