import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def create_directory_structure(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join("annotation_check", "train"), exist_ok=True)
    os.makedirs(os.path.join("annotation_check", "test"), exist_ok=True)
    os.makedirs(os.path.join("annotation_check", "val"), exist_ok=True)

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def update_ids(data, start_id):
    new_images = []
    new_annotations = []
    image_id_map = {}
    
    print(f"Updating IDs. Starting ID: {start_id}")
    print(f"Original image count: {len(data['images'])}")
    print(f"Original annotation count: {len(data['annotations'])}")

    for image in data['images']:
        new_image_id = start_id
        image_id_map[image['id']] = new_image_id
        image['id'] = new_image_id
        new_images.append(image)
        start_id += 1

    for annotation in data['annotations']:
        new_annotation_id = start_id
        if annotation['image_id'] in image_id_map:
            annotation['id'] = new_annotation_id
            annotation['image_id'] = image_id_map[annotation['image_id']]
            new_annotations.append(annotation)
            start_id += 1
        else:
            print(f"Warning: annotation {annotation['id']} refers to non-existent image {annotation['image_id']}")

    print(f"Updated image count: {len(new_images)}")
    print(f"Updated annotation count: {len(new_annotations)}")
    print(f"New starting ID: {start_id}")

    data['images'] = new_images
    data['annotations'] = new_annotations
    return data, start_id

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def copy_images(source_dir, dest_dir):
    images = os.listdir(source_dir)
    for image in images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(dest_dir, image))

def normalize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    return [x/img_width, y/img_height, w/img_width, h/img_height]

def denormalize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    return [int(x*img_width), int(y*img_height), int(w*img_width), int(h*img_height)]

def plot_annotations(image_dir, json_data, save_dir):
    print(f"Plotting annotations for {len(json_data['images'])} images")
    for img_data in tqdm(json_data['images']):
        img_path = os.path.join(image_dir, img_data['file_name'])
        img = cv2.imread(img_path)
        
        annotations = [ann for ann in json_data['annotations'] if ann['image_id'] == img_data['id']]
        print(f"Image {img_data['file_name']} has {len(annotations)} annotations")
        
        for ann in annotations:
            bbox = ann['bbox']
            print(f"Original bbox: {bbox}")
            
            # Denormalize if necessary
            if max(bbox) <= 1:
                bbox = denormalize_bbox(bbox, img_data['width'], img_data['height'])
                print(f"Denormalized bbox: {bbox}")
            
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        save_path = os.path.join(save_dir, img_data['file_name'])
        cv2.imwrite(save_path, img)

def combine_data(data1, data2):
    combined_data = {
        "info": data1['info'],
        "licenses": data1['licenses'],
        "images": data1['images'] + data2['images'],
        "annotations": data1['annotations'] + data2['annotations'],
        "categories": data1['categories']
    }
    
    print(f"Combined data: {len(combined_data['images'])} images, {len(combined_data['annotations'])} annotations")
    return combined_data

def main(folder1, folder2):
    combined_data_dir = "combined_data"
    annotation_check_dir = "annotation_check"
    train_dir = os.path.join(combined_data_dir, "train")
    test_dir = os.path.join(combined_data_dir, "test")
    val_dir = os.path.join(combined_data_dir, "val")

    create_directory_structure(combined_data_dir)

    train_data1 = read_json(os.path.join(folder1, 'train', 'train_split.json'))
    test_data1 = read_json(os.path.join(folder1, 'test', 'test_split.json'))
    val_data1 = read_json(os.path.join(folder1, 'val', 'val_split.json'))

    train_data2 = read_json(os.path.join(folder2, 'train', 'train_split.json'))
    test_data2 = read_json(os.path.join(folder2, 'test', 'test_split.json'))
    val_data2 = read_json(os.path.join(folder2, 'val', 'val_split.json'))

    start_id = 1
    print("Updating IDs for folder1 data")
    train_data1, start_id = update_ids(train_data1, start_id)
    test_data1, start_id = update_ids(test_data1, start_id)
    val_data1, start_id = update_ids(val_data1, start_id)

    print("\nUpdating IDs for folder2 data")
    train_data2, start_id = update_ids(train_data2, start_id)
    test_data2, start_id = update_ids(test_data2, start_id)
    val_data2, start_id = update_ids(val_data2, start_id)

    print("\nCombining data")
    combined_train_data = combine_data(train_data1, train_data2)
    combined_test_data = combine_data(test_data1, test_data2)
    combined_val_data = combine_data(val_data1, val_data2)

    save_json(combined_train_data, os.path.join(train_dir, "train_split.json"))
    save_json(combined_test_data, os.path.join(test_dir, "test_split.json"))
    save_json(combined_val_data, os.path.join(val_dir, "val_split.json"))

    copy_images(os.path.join(folder1, 'train', 'images'), os.path.join(train_dir, "images"))
    copy_images(os.path.join(folder2, 'train', 'images'), os.path.join(train_dir, "images"))
    copy_images(os.path.join(folder1, 'test', 'images'), os.path.join(test_dir, "images"))
    copy_images(os.path.join(folder2, 'test', 'images'), os.path.join(test_dir, "images"))
    copy_images(os.path.join(folder1, 'val', 'images'), os.path.join(val_dir, "images"))
    copy_images(os.path.join(folder2, 'val', 'images'), os.path.join(val_dir, "images"))

    print("\nPlotting annotations for training data")
    plot_annotations(os.path.join(train_dir, "images"), combined_train_data, os.path.join(annotation_check_dir, "train"))
    print("\nPlotting annotations for test data")
    plot_annotations(os.path.join(test_dir, "images"), combined_test_data, os.path.join(annotation_check_dir, "test"))
    print("\nPlotting annotations for validation data")
    plot_annotations(os.path.join(val_dir, "images"), combined_val_data, os.path.join(annotation_check_dir, "val"))

    print("Data combination and annotation plotting complete.")

# Example usage
folder1 = '/home/frinksserver/Deepak/OCR/datasets/skh/skh_ocr_fasterrcnn/split_june4'
folder2 = '/home/frinksserver/Deepak/OCR/data_preparation/fasterrcnn_ocr_dataprep/split'
main(folder1, folder2)