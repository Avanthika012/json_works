import os
import json
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def plot_contours(image_path, annotations, output_path):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    contour_count = 0
    for annotation in annotations:
        if annotation['segmentation']:
            points = annotation['segmentation'][0]
            xy = list(zip(points[0::2], points[1::2]))
            draw.polygon(xy, outline=(255, 0, 0))
            contour_count += 1
    
    # Create a black canvas on the right side
    canvas_width = 200
    new_image = Image.new('RGB', (image.width + canvas_width, image.height), color='black')
    new_image.paste(image, (0, 0))
    
    # Add text to the black canvas
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()
    text = f"Contours: {contour_count}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = (image.width + (canvas_width - text_width) // 2, (image.height - text_height) // 2)
    draw.text(text_position, text, fill=(255, 255, 255), font=font)
    
    new_image.save(output_path)
    print(f'Contours plotted and saved to {output_path}')

def process_annotations(root_folder):
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "biscuit", "supercategory": ""}]
    }
    image_id_counter = 1
    annotation_id_counter = 1

    combined_data_folder = os.path.join(os.path.dirname(root_folder), "combined_data")
    combined_images_folder = os.path.join(combined_data_folder, "images")
    os.makedirs(combined_images_folder, exist_ok=True)

    annotation_check_folder = os.path.join(os.path.dirname(root_folder), "annotation_check")
    os.makedirs(annotation_check_folder, exist_ok=True)

    ignore_images = ["1817_cropped_1437.png", "1039_cropped_1838.png"]

    for folder in tqdm(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path) and "biscuit" in folder:
            json_path = os.path.join(folder_path, "annotations", "instances_default.json")
            with open(json_path) as f:
                data = json.load(f)

            annotation_check_subfolder = os.path.join(annotation_check_folder, folder)
            os.makedirs(annotation_check_subfolder, exist_ok=True)

            for image_info in tqdm(data['images'], desc=f"Processing images in {folder}"):
                if image_info['file_name'] in ignore_images:
                    continue  # Skip ignored images

                image_path = os.path.join(folder_path, "images", image_info['file_name'])
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_info['file_name']}")
                    continue

                annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_info['id']]
                plot_contours(image_path, annotations, os.path.join(annotation_check_subfolder, image_info['file_name']))

            # Add images and update image IDs
            for image_info in data['images']:
                if image_info['file_name'] in ignore_images:
                    continue  # Skip ignored images

                new_image_info = image_info.copy()
                old_image_id = image_info['id']
                new_image_info['id'] = image_id_counter
                combined_data['images'].append(new_image_info)

                src_image_path = os.path.join(folder_path, "images", image_info['file_name'])
                dst_image_path = os.path.join(combined_images_folder, image_info['file_name'])
                shutil.copy(src_image_path, dst_image_path)

                # Update annotations with new image ID
                for ann in data['annotations']:
                    if ann['image_id'] == old_image_id:
                        new_ann = ann.copy()
                        new_ann['id'] = annotation_id_counter
                        new_ann['image_id'] = image_id_counter
                        combined_data['annotations'].append(new_ann)
                        annotation_id_counter += 1

                image_id_counter += 1

    combined_json_path = os.path.join(combined_data_folder, "combined_json.json")
    with open(combined_json_path, 'w') as f:
        json.dump(combined_data, f)

    annotation_check_combined = os.path.join(annotation_check_folder, "combined_data")
    os.makedirs(annotation_check_combined, exist_ok=True)

    for image_info in tqdm(combined_data['images'], desc="Plotting combined annotations"):
        if image_info['file_name'] in ignore_images:
            continue  # Skip ignored images

        image_path = os.path.join(combined_images_folder, image_info['file_name'])
        annotations = [ann for ann in combined_data['annotations'] if ann['image_id'] == image_info['id']]
        plot_contours(image_path, annotations, os.path.join(annotation_check_combined, image_info['file_name']))

# Ensure the paths are correctly updated to match your environment
# process_annotations("/path/to/root_folder")

process_annotations("biscuits_data")
