import json
import os
import shutil
from sklearn.model_selection import train_test_split

# Load COCO annotations
coco_annotation_path = '/home/frinksserver/Deepak/OCR/datasets/skh/skh_ocr_fasterrcnn/july23_skhocr/instances_default.json'  # Update this path
with open(coco_annotation_path) as f:
    coco_dataset = json.load(f)

# Extract image IDs
image_ids = [img['id'] for img in coco_dataset['images']]

# Split the dataset
train_ids, test_val_ids = train_test_split(image_ids, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)

def filter_annotations(image_ids, annotations):
    """Filters annotations to include only those that correspond to the given image IDs."""
    return [annotation for annotation in annotations if annotation['image_id'] in image_ids]

def create_coco_split(image_ids, coco_dataset, split_name, target_dir):
    """Creates a COCO-format JSON file for the specified split and saves it to the target directory."""
    images = [img for img in coco_dataset['images'] if img['id'] in image_ids]
    annotations = filter_annotations(image_ids, coco_dataset['annotations'])
    split_dataset = {
        "info": coco_dataset['info'],
        "licenses": coco_dataset['licenses'],
        "images": images,
        "annotations": annotations,
        "categories": coco_dataset['categories']
    }
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    json_file_path = os.path.join(target_dir, f'{split_name}_split.json')
    with open(json_file_path, 'w') as f:
        json.dump(split_dataset, f)

def move_images(image_ids, source_dir, target_dir, coco_dataset):
    target_dir = target_dir+"/images"
    """Copies images from the source directory to the target directory based on the specified image IDs."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    image_files = [img['file_name'] for img in coco_dataset['images'] if img['id'] in image_ids]
    for file_name in image_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(source_path, target_path)  # Use shutil.move to move instead of copy

# Define directories
source_directory = '/home/frinksserver/Deepak/OCR/datasets/skh/skh_ocr_fasterrcnn/july23_skhocr/images'
train_directory = './split/train'
val_directory = './split/val'
test_directory = './split/test'

# Create JSON files for each split and move/copy the images
create_coco_split(train_ids, coco_dataset, 'train', train_directory)
create_coco_split(val_ids, coco_dataset, 'val', val_directory)
create_coco_split(test_ids, coco_dataset, 'test', test_directory)

move_images(train_ids, source_directory, train_directory, coco_dataset)
move_images(val_ids, source_directory, val_directory, coco_dataset)
move_images(test_ids, source_directory, test_directory, coco_dataset)
