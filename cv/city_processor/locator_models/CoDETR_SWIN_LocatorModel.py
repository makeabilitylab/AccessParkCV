import os
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO
from utils import deg2num, num2deg
from lrucache import LRUCache
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box


class DisabilityParkingSpaceLocatorCODetrSWIN:

    def __init__(self, config_path, model_path, confidence_threshold=0.3, imgsz=512):

        self.confidence_threshold = confidence_threshold

        # Models
        self.config_path = config_path
        self.locator_model_path = model_path
        self.imgsz = imgsz

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.locator_model = None
        
        self.open_model()

    def open_model(self):
        self.locator_model = init_detector(self.config_path, self.locator_model_path, device=self.device)

    def filter_overlapping_by_confidence(self, detections, iou_threshold=0.6):
        """
        Filter overlapping polygon detections, keeping only the one with higher confidence
        when IoU exceeds the threshold.
        
        Args:
            detections: List of detection dictionaries with 'conf' and 'xyxyxyxy' keys
            iou_threshold: IoU threshold above which to consider detections as overlapping
            
        Returns:
            List of filtered detections
        """
        if not detections:
            return []
        
        # Sort detections by confidence score (descending)
        sorted_detections = sorted(detections, key=lambda x: x['bbox_conf'], reverse=True)
        
        # Convert polygon coordinates to Shapely Polygon objects
        polygons = []
        for detection in sorted_detections:
            bbox = detection['bbox']
            # coords = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
            # print(bbox)
            # poly = Polygon(bbox)
            poly = box(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
            polygons.append(poly)
        
        # Initialize list to keep track of indices to keep
        indices_to_keep = []
        
        for i in range(len(sorted_detections)):
            should_keep = True
            
            # Check if current detection overlaps with any higher confidence detection
            for j in indices_to_keep:
                # Calculate IoU
                intersection_area = polygons[i].intersection(polygons[j]).area
                union_area = polygons[i].area + polygons[j].area - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0
                
                if iou > iou_threshold:
                    should_keep = False
                    break
                    
            if should_keep:
                indices_to_keep.append(i)
        
        # Return filtered detections
        return [sorted_detections[i] for i in indices_to_keep]

    def detect(self, image, tile_x=None, tile_y=None):
        """
        Runs the locator model on an image. Returns a list of dictionaries containing detected objects:

        [
            {
                'class_name': dp_one_aisle,
                'bbox': (x1, y1, x2, y2),
                'polygon': (x1, y1, x2, y2, x3, y3) # if available
            }
            {
            ...
            }
        ]
        """
        image = np.array(image)
        result = inference_detector(self.locator_model, image)

        detection_threshold = self.confidence_threshold # confidence
        bbox_result = result
        # if instance segmentation
        # bbox_result, segm_result = result
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)\
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        labels_impt = np.where(bboxes[:, -1] > detection_threshold)[0]

        classes = ('access_aisle','curbside','dp_no_aisle','dp_one_aisle','dp_two_aisle','one_aisle','two_aisle')
        labels_impt_list = [labels[i] for i in labels_impt]
        labels_class = [classes[i] for i in labels_impt_list]

        detected_objects = []
        for i, detected_obj_idx in enumerate(labels_impt):
            left = bboxes[detected_obj_idx][0]
            top = bboxes[detected_obj_idx][1]
            right = bboxes[detected_obj_idx][2]
            bottom = bboxes[detected_obj_idx][3]
            
            predicted_class = labels_class[i]

            detected_objects.append(
                    {'class_name': predicted_class,
                     'bbox': ((left, top), (right, bottom)),
                     'bbox_conf': float(bboxes[detected_obj_idx][-1])})

        detected_objects = self.filter_overlapping_by_confidence(detected_objects, iou_threshold=0.6)

        return detected_objects
