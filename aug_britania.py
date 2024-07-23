import os
import json
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import shutil

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def augment_dataset(input_folder, input_annotation, output_folder, split_type):
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    
    with open(input_annotation, 'r') as f:
        annotations = json.load(f)
    
    augmentations = {
        'randbc': A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=1),
        'gaussblur': A.GaussianBlur(blur_limit=(3, 7), p=1)
    }
    
    new_annotations = annotations.copy()
    new_annotations['images'] = []
    new_annotations['annotations'] = []
    image_id = 0
    annotation_id = 0
    
    for img in tqdm(annotations['images'], desc="Processing original images"):
        src_path = os.path.join(input_folder, img['file_name'])
        dst_path = os.path.join(output_folder, 'images', img['file_name'])
        cv2.imwrite(dst_path, cv2.imread(src_path))
        
        new_img_info = img.copy()
        new_img_info['id'] = image_id
        new_annotations['images'].append(new_img_info)
        
        for ann in annotations['annotations']:
            if ann['image_id'] == img['id']:
                new_ann = ann.copy()
                new_ann['id'] = annotation_id
                new_ann['image_id'] = image_id
                new_annotations['annotations'].append(new_ann)
                annotation_id += 1
        
        image_id += 1
    
    for img in tqdm(annotations['images'], desc="Augmenting images"):
        img_path = os.path.join(input_folder, img['file_name'])
        image = cv2.imread(img_path)
        
        for aug_name, aug_transform in augmentations.items():
            augmented = aug_transform(image=image)
            aug_image = augmented['image']
            
            base_name, ext = os.path.splitext(img['file_name'])
            new_file_name = f"{base_name}_aug_{aug_name}{ext}"
            
            output_path = os.path.join(output_folder, 'images', new_file_name)
            cv2.imwrite(output_path, aug_image)
            
            new_img_info = img.copy()
            new_img_info['id'] = image_id
            new_img_info['file_name'] = new_file_name
            new_annotations['images'].append(new_img_info)
            
            for ann in annotations['annotations']:
                if ann['image_id'] == img['id']:
                    new_ann = ann.copy()
                    new_ann['id'] = annotation_id
                    new_ann['image_id'] = image_id
                    new_annotations['annotations'].append(new_ann)
                    annotation_id += 1
            
            image_id += 1
    
    output_annotation = os.path.join(output_folder, f'{split_type}_split.json')
    save_json(new_annotations, output_annotation)
    print(f"Augmentation complete. Total images: {len(new_annotations['images'])}")

def create_folder_structure(base_folder):
    folders = [
        'aug_train_test_split/train/images',
        'aug_train_test_split/test/images',
        'annotation_check/aug_train',
        'annotation_check/aug_test'
    ]
    for folder in tqdm(folders, desc="Creating folders"):
        os.makedirs(os.path.join(base_folder, folder), exist_ok=True)

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
    
    canvas_width = 200
    new_image = Image.new('RGB', (image_width + canvas_width, image_height), color='black')
    new_image.paste(image, (0, 0))
    
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()
    text = f"Contours: {contour_count}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = (image_width + (canvas_width - text_width) // 2, (image_height - text_height) // 2)
    draw.text(text_position, text, fill=(255, 255, 255), font=font)
    
    print(f"Attempting to save image to: {output_path}")
    try:
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
    base_folder = '.'
    create_folder_structure(base_folder)
    
    train_input_folder = os.path.join(base_folder, 'train_test_split/train/images')
    train_input_annotation = os.path.join(base_folder, 'train_test_split/train/train.json')
    train_output_folder = os.path.join(base_folder, 'aug_train_test_split/train')
    
    test_input_folder = os.path.join(base_folder, 'train_test_split/test/images')
    test_input_annotation = os.path.join(base_folder, 'train_test_split/test/test.json')
    test_output_folder = os.path.join(base_folder, 'aug_train_test_split/test')
    
    augment_dataset(train_input_folder, train_input_annotation, train_output_folder, 'train')
    augment_dataset(test_input_folder, test_input_annotation, test_output_folder, 'test')
    
    train_data = load_json(os.path.join(train_output_folder, 'train_split.json'))
    test_data = load_json(os.path.join(test_output_folder, 'test_split.json'))
    
    plot_annotations(train_data, os.path.join(base_folder, 'aug_train_test_split/train/images'), os.path.join(base_folder, 'annotation_check/aug_train'))
    plot_annotations(test_data, os.path.join(base_folder, 'aug_train_test_split/test/images'), os.path.join(base_folder, 'annotation_check/aug_test'))
    
    print("Augmentation and annotation plotting completed successfully!")

if __name__ == "__main__":
    main()
