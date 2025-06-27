import torch
import cv2
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = True

datapath = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/YOLOLarger/data.yaml'

# Load a COCO-pretrained YOLOv8n model
model = YOLO('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/YOLO/runs/largerdataset/detect/train/weights/best.pt') # detect medium

# Display model information (optional)
model.info()

class_idx_to_name = ['access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']

# output path
output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/YOLO/evaluation_results/yolo_predictions_testset.json'

detection_threshold = 0.3 # confidence
def predict_with_model(file_path):

    image = Image.open(file_path)
    results = model(image, verbose=False)
    detected_objects = []
    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = float(x1); y1 = float(y1)
            x2 = float(x2); y2 = float(y2)
            
            predicted_class = class_idx_to_name[int(box.cls)]

            detected_objects.append(
                    {'category_id': predicted_class,
                    'bbox': str((x1, y1, x2, y2)),
                    'score': str(float(box.conf))})

    img_name = os.path.basename(file_path)
    return img_name, detected_objects

def process_images_in_folder(folder_path):
    """
    Applies a specified function to every image in a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing the images.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path '{folder_path}' is not valid.")

    output = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"Processing {filename}")
            basename, detected_objects = predict_with_model(file_path)
            output[basename] = detected_objects
        except Exception as e:
            print(f"Skipping {filename}. Error: {e}")

    with open(output_path, 'w') as f:
            json.dump(output, f)

process_images_in_folder('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/YOLOLarger/test_no_aa/images')