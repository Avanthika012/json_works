'''
This code plots boxes from json file created by cvat for fasterrcnn on the corrsponding images
input: 
    json_file_path = "instances_default_converted.json"
    images_dir = "/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/train"
out:
    output_dir = "/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/train_plot_json"
'''

import json
import os
import cv2
from tqdm import tqdm

def plot_boxes_on_images(json_file_path, images_dir, output_dir):
    # Load the COCO-formatted JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create a dictionary to map image IDs to their file paths
    image_id_to_file_path = {img['id']: img['file_name'] for img in data['images']}
    
    # Create a dictionary to hold annotations for each image
    image_id_to_annotations = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for image in tqdm(data['images'], desc="Processing images"):
        image_id = image['id']
        image_file_path = os.path.join(images_dir, image_id_to_file_path[image_id])
        
        # Load the image
        img = cv2.imread(image_file_path)
        
        if img is None:
            print(f"Image not found: {image_file_path}")
            continue
        
        # Draw each bounding box
        if image_id in image_id_to_annotations:
            for annotation in image_id_to_annotations[image_id]:
                bbox = annotation['bbox']
                x, y, width, height = bbox
                start_point = (int(x), int(y))
                end_point = (int(x + width), int(y + height))
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2
                img = cv2.rectangle(img, start_point, end_point, color, thickness)
        
        # Save the image with bounding boxes
        output_image_path = os.path.join(output_dir, os.path.basename(image_file_path))
        cv2.imwrite(output_image_path, img)

# Paths to the JSON file, images directory, and output directory
json_file_path = "instances_default_converted.json"
images_dir = "/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/train"
output_dir = "/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/train_plot_json"

# Plot the boxes on the images and save them
plot_boxes_on_images(json_file_path, images_dir, output_dir)
