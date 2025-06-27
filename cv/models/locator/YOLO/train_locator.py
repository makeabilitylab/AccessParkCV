import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = True

# datapath = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/data_augmentation/DisabilityParkingCV.v5i.yolo/data.yaml'
# project = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/YOLO/runs/detect'

# larger dataset
datapath = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/YOLOLarger/data.yaml'
project = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/YOLO/runs/largerdataset/detect'

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolo11m.pt") # detect medium

# Display model information (optional)
model.info()
results = model.train(data=datapath,
    epochs=200,
    imgsz=512,
    plots=True,
    verbose=verbose,
    project=project,
    device=device)
    # ,
    # # Disable all augmentation
    # mosaic=0.0,
    # mixup=0.0,
    # copy_paste=0.0,
    # hsv_h=0.0,
    # hsv_s=0.0,
    # hsv_v=0.0,
    # degrees=0.0,
    # translate=0.0,
    # scale=0.0,
    # shear=0.0,
    # perspective=0.0,
    # flipud=0.0,
    # fliplr=0.0,
    # auto_augment=None,
    # erasing=0.0)
