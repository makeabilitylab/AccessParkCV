import os, sys
import torch, json
import numpy as np
from tqdm import tqdm

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

# model_config_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/config/DINO/DINO_5scale_disabilityparking.py" # change the path of the model config file
# model_checkpoint_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/logs/DINO/dp-R50-MS4/checkpoint0035.pth" # change the path of the model checkpoint
# output path
# output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/logs/DINO/dp-R50-MS4/dino_predictions_36epoch.json'

model_config_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/config/DINO/DINO_5scale_disabilityparking_largerdataset.py" # change the path of the model config file
model_checkpoint_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/logs/DINO/dp-R50-MS4-largerdataset/checkpoint0027.pth" # change the path of the model checkpoint
output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/logs/DINO/dp-R50-MS4-largerdataset/dino_predictions_27epoch_aug.json'


# Image size
image_size = 512

id2name = ['objects', 'access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']


args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

args.dataset_file = 'coco'
args.coco_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DatasetLarger" # the path of coco
args.fix_size = False

dataset_test = build_dataset(image_set='test', args=args)   



def predict_with_model(filename, image, targets):
    # Model output
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    threshold = 0.3 # set a threshold
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']
    # boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > threshold

    above_threshold_boxes = list(boxes[select_mask])
    for i in range(len(above_threshold_boxes)):
        x1, y1, x2, y2 = above_threshold_boxes[i]
        abs_x1 = float(x1 * image_size)
        abs_y1 = float(y1 * image_size)
        abs_x2 = float(x2 * image_size)
        abs_y2 = float(y2 * image_size)
        above_threshold_boxes[i] = [abs_x1, abs_y1, abs_x2, abs_y2]

    above_threshold_labels = [id2name[int(item)] for item in labels[select_mask]]
    above_threshold_scores = scores[select_mask]

    detected_objects = []
    for i in range(len(above_threshold_boxes)):
        detected_objects.append(
            {'category_id': above_threshold_labels[i],
            'bbox': str(above_threshold_boxes[i]),
            'score': str(float(above_threshold_scores[i]))})

    # if len(detected_objects) > 0:
    #     import cv2 
    #     filepath = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DisabilityParkingCV.v5i.coco/test2017/" + filename
    #     cv2_image = cv2.imread(filepath)
    #     color=(0,255,0)
    #     for box in above_threshold_boxes:
    #         x1, y1, x2, y2 = map(int, box)
    #         cv2.rectangle(cv2_image, (x1, y1), (x2, y2), color, 2)
    #     cv2.imwrite('test.jpg', cv2_image)
    #     exit()

    img_name = os.path.basename(filename)
    return img_name, detected_objects


def process_test_images_in_dataset():
    """
    Applies a specified function to every image in a given folder.

    """

    output = {}
    for i, (image, targets) in enumerate(tqdm(dataset_test)):
        img_id = dataset_test.ids[i]
        filename = dataset_test.coco.loadImgs(img_id)[0]['file_name']
        
        try:
            basename, detected_objects = predict_with_model(filename, image, targets)
            output[basename] = detected_objects
        except Exception as e:
            print(f"Skipping {filename}. Error: {e}")

    with open(output_path, 'w') as f:
            json.dump(output, f)

process_test_images_in_dataset()