import os
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO
from utils import deg2num, num2deg
from lrucache import LRUCache


class DisabilityParkingSpaceLocatorYOLO:

    def __init__(self, model_path, imgsz=512):

        # Models
        self.locator_model_path = model_path
        self.imgsz = imgsz

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.locator_model = None
        
        self.open_model()

    def open_model(self):
        self.locator_model = YOLO(self.locator_model_path)
        self.locator_model.to(self.device)

    def detect(self, image, tile_x=None, tile_y=None):
        """
        Runs the locator model on an image. Returns a list of dictionaries containing detected objects:

        [{
            'class_name': dp_one_aisle,
            'bbox': (x1, y1, x2, y2),
            'polygon': (x1, y1, x2, y2, x3, y3) # if available
        }]
        
        
        Returns a list containing detected objects:
        [((bbox coords), class_name), ...]
        """
        results = self.locator_model(image, verbose=False)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                predicted_class = self.locator_model.names[int(box.cls)]
                detected_objects.append(
                    {'class_name': predicted_class,
                     'bbox': (x1, y1, x2, y2)})
        return detected_objects
