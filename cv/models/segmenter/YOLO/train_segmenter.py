import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = True

datapath = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/dataset_made_from_crops/YOLO_larger/data.yaml'
project = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb'

# Load a COCO-pretrained YOLOv8n model
# model = YOLO("yolo11l-obb.pt") # detect medium
model = YOLO("yolo11x-obb.pt") # detect medium

# Display model information (optional)
model.info()
results = model.train(data=datapath,
    epochs=200,
    imgsz=100,
    plots=True,
    verbose=verbose,
    project=project,
    device=device)
    # ,
    # mosaic=0.0,
    # # mixup=0.0,
    # # copy_paste=0.0,
    # # hsv_h=0.0,
    # # hsv_s=0.0,
    # # hsv_v=0.0,
    # # degrees=0.0,
    # translate=0.0,
    # scale=0.0,
    # # shear=0.0,
    # # perspective=0.0,
    # flipud=0.0,
    # fliplr=0.0,
    # # auto_augment=None,
    # erasing=0.0)

# comment out the ones you want