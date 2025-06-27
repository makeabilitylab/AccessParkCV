import torch
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = True

datapath = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/training_data/DisabilityParkingCV.v4i.yolov11/data.yaml'
project = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/runs/detect'

# Load a COCO-pretrained YOLOv8n model
model = YOLO('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/runs/detect/train3/weights/best.pt') # detect medium

# Display model information (optional)
model.info()

class_idx_to_name = ['access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']


def locate_objects(file_path):
    image = Image.open(file_path)
    results = model(image, verbose=False)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            predicted_class = class_idx_to_name[int(box.cls)]
            detected_objects.append([predicted_class, x1, y1, x2, y2])
    return detected_objects

def process_image_and_labels(base_name, image_path, label_path, output_dir):
    image = cv2.imread(image_path)
    image_with_bboxes = image.copy()
    height, width, _ = image.shape
    
    # Define a color map for the 7 classes (in BGR format)
    color_map = {
        'access_aisle': (0, 168, 255),      
        'curbside': (0, 255, 5),     
        'dp_no_aisle': (0, 41, 255),       
        'dp_one_aisle': (0, 140, 255),     
        'dp_two_aisle': (0, 240, 255),     
        'one_aisle': (235, 0, 0),    
        'two_aisle': (255, 157, 157)    
    }
    for key, val in color_map.items():
        temp = (val[2], val[1], val[0])
        color_map[key] = temp
    
    # Default color for any unexpected classes
    default_color = (252, 15, 3)  # Original color used
    
    # Read text file
    with open(label_path, 'r') as file:
        lines = file.readlines()
    if len(lines) == 0:
        return
    
    # Parse and draw polygons
    for line in lines:
        data = line.strip().split()
        cls = class_idx_to_name[int(data[0])]
        if cls == 'access_aisle':
            continue
        
        # Get color for this class
        color = color_map.get(cls, default_color)
        
        points = list(map(float, data[1:]))
        # Convert normalized coordinates to pixel coordinates
        pixel_points = [(int(points[i] * width), int(points[i+1] * height)) for i in range(0, len(points), 2)]
        pixel_points = np.array(pixel_points, np.int32)
        pixel_points = pixel_points.reshape((-1, 1, 2))
        
        # Draw polygon border with class-specific color
        image = cv2.polylines(image, [pixel_points], isClosed=True, color=color, thickness=1)
        
        # Add smaller text within bounding box with matching color
        x, y, w, h = cv2.boundingRect(pixel_points)
        font_scale = max(0.3, min(w / len(cls) * 0.05, 1))
        cv2.putText(image, cls, (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    
    bbox_lines = locate_objects(image_path)
    # Parse and draw bounding boxes and class names
    for line in bbox_lines:
        cls = line[0]
        if cls == 'access_aisle':
            continue
        
        # Get color for this class
        color = color_map.get(cls, default_color)
        
        x1, y1, x2, y2 = map(int, line[1:5])
        # Draw bounding box with class-specific color
        image_with_bboxes = cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), color=color, thickness=1)
        
        # Add smaller text within bounding box with matching color
        font_scale = max(0.3, min((x2 - x1) / len(cls) * 0.05, 1))
        cv2.putText(image_with_bboxes, cls, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    
    # Add "Ground Truth" and "Predicted" labels on top left of each image
    # Parameters for the title text
    title_font_scale = 0.8
    title_thickness = 2
    title_color = (0, 0, 255)  # Red in BGR
    title_position_y = 30  # Position from top
    
    # Add "Ground Truth" to the left image
    cv2.putText(image, "Ground Truth", (10, title_position_y), 
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_color, title_thickness, cv2.LINE_AA)
    
    # Add "Predicted" to the right image
    cv2.putText(image_with_bboxes, "Predicted", (10, title_position_y), 
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_color, title_thickness, cv2.LINE_AA)
    
    # Combine the two images side by side
    combined_image = np.hstack((image, image_with_bboxes))
    
    # Save the resulting image
    output_path = os.path.join(output_dir, f'{base_name}.jpg')
    cv2.imwrite(output_path, combined_image)

def main():
    # Define directories
    image_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/training_data/DisabilityParkingCV.v4i.yolov11/valid/images/'
    label_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/training_data/DisabilityParkingCV.v4i.yolov11/valid/labels/'
    output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/validation_output'
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files (supporting common image formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    
    # Process each image
    for image_path in tqdm(image_files):
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Look for corresponding text file
        text_path = os.path.join(label_dir, f"{base_name}.txt")
        if os.path.exists(text_path):
            process_image_and_labels(base_name, image_path, text_path, output_dir)
        else:
            print(f"Warning: No matching text file found for {image_path}")

if __name__ == "__main__":
    main()
