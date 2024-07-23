import json
import os
import shutil
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def verify_images(data, image_folder):
    img_not_found = []
    print("Verifying images...")
    for image in tqdm(data['images'], desc="Checking images"):
        if not os.path.exists(os.path.join(image_folder, image['file_name'])):
            print(f"Image not found: {image['file_name']}")
            img_not_found.append(image['file_name'])
    
    if not img_not_found:
        print("All images found successfully!")
    return img_not_found

def split_dataset(data, train_ratio=0.8):
    images = data['images']
    random.shuffle(images)
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    train_data = {k: v for k, v in data.items() if k != 'images'}
    test_data = {k: v for k, v in data.items() if k != 'images'}
    
    train_data['images'] = train_images
    test_data['images'] = test_images
    
    print("Splitting annotations...")
    train_data['annotations'] = [ann for ann in tqdm(data['annotations'], desc="Processing train annotations") 
                                 if ann['image_id'] in [img['id'] for img in train_images]]
    test_data['annotations'] = [ann for ann in tqdm(data['annotations'], desc="Processing test annotations") 
                                if ann['image_id'] in [img['id'] for img in test_images]]
    
    return train_data, test_data

def create_folder_structure():
    folders = [
        'train_test_split/train/images',
        'train_test_split/test/images',
        'annotation_check/train',
        'annotation_check/test'
    ]
    for folder in tqdm(folders, desc="Creating folders"):
        os.makedirs(folder, exist_ok=True)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def copy_images(data, src_folder, dest_folder):
    for image in tqdm(data['images'], desc=f"Copying images to {dest_folder}"):
        shutil.copy(
            os.path.join(src_folder, image['file_name']),
            os.path.join(dest_folder, image['file_name'])
        )

def plot_contours(image_path, annotations, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    image_width, image_height = image.size
    contour_count = 0
    
    for annotation in annotations:
        points = annotation['segmentation'][0]
        if len(points) % 2 == 0:
            xy = list(zip(points[0::2], points[1::2]))
            xy = [(max(0, min(x, image_width)), max(0, min(y, image_height))) for x, y in xy]
            draw.line(xy + [xy[0]], fill=(255, 0, 0), width=2)
            contour_count += 1
        print(f"Plotted contour with {len(xy)} points")
    
    # Create a black canvas on the right side
    canvas_width = 200
    new_image = Image.new('RGB', (image_width + canvas_width, image_height), color='black')
    new_image.paste(image, (0, 0))
    
    # Add text to the black canvas
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()
    text = f"Contours: {contour_count}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = (image_width + (canvas_width - text_width) // 2, (image_height - text_height) // 2)
    draw.text(text_position, text, fill=(255, 255, 255), font=font)
    
    print(f"Attempting to save image to: {output_path}")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_image.save(output_path)
        print(f'Contours plotted and saved to {output_path}')
    except Exception as e:
        print(f"Error saving image: {e}")

def plot_annotations(data, image_folder, output_folder):
    image_dict = {img['id']: img for img in data['images']}
    annotations_by_image_id = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)
    
    for image_id, annotations in tqdm(annotations_by_image_id.items(), desc=f"Plotting annotations for {output_folder}"):
        img_info = image_dict[image_id]
        image_path = os.path.join(image_folder, img_info['file_name'])
        output_path = os.path.join(output_folder, f"annotated_{img_info['file_name']}")
        plot_contours(image_path, annotations, output_path)
        print(f"Plotted {len(annotations)} annotations for image {img_info['file_name']}")

def main():
    print("Starting image annotation processing...")
    
    # Load JSON data
    data = load_json('combined_data/combined_json.json')
    print("JSON data loaded successfully.")
    
    # Verify images
    img_not_found = verify_images(data, 'combined_data/images')
    if img_not_found:
        print("Missing images:", img_not_found)
        return
    
    # Split dataset
    print("Splitting dataset...")
    train_data, test_data = split_dataset(data)
    print("Dataset split completed.")
    
    # Create folder structure
    print("Creating folder structure...")
    create_folder_structure()
    
    # Save split data
    print("Saving split data...")
    save_json(train_data, 'train_test_split/train/train.json')
    save_json(test_data, 'train_test_split/test/test.json')
    print("Split data saved.")
    
    # Copy images
    print("Copying images...")
    copy_images(train_data, 'combined_data/images', 'train_test_split/train/images')
    copy_images(test_data, 'combined_data/images', 'train_test_split/test/images')
    
    # Plot annotations
    print("Plotting annotations...")
    plot_annotations(train_data, 'train_test_split/train/images', 'annotation_check/train')
    plot_annotations(test_data, 'train_test_split/test/images', 'annotation_check/test')
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()
