import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

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

# Example usage:
def main():
    base_folder = '.'
    
    # Load the JSON data
    train_input_annotation = os.path.join(base_folder, 'aug_train_test_split/train/train_split.json')
    train_data = load_json(train_input_annotation)
    test_input_annotation = os.path.join(base_folder, 'aug_train_test_split/test/test_split.json')
    test_data = load_json(test_input_annotation)

    
    # Define input image folder and output folder for annotated images
    train_input_folder = os.path.join(base_folder, 'aug_train_test_split/train/images')
    train_output_folder = os.path.join(base_folder, 'annotation_check/custom_aug_train')

    # Define input image folder and output folder for annotated images
    test_input_folder = os.path.join(base_folder, 'aug_train_test_split/test/images')
    test_output_folder = os.path.join(base_folder, 'annotation_check/custom_aug_test')
    
    # Plot and save annotated images
    plot_annotations(train_data, train_input_folder, train_output_folder)
    plot_annotations(test_data, test_input_folder, test_output_folder)

if __name__ == "__main__":
    main()
