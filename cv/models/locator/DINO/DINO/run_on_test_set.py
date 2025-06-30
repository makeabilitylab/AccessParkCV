import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/config/DINO/DINO_5scale_disabilityparking.py" # change the path of the model config file
model_checkpoint_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/logs/DINO/dp-R50-MS4/checkpoint0035.pth" # change the path of the model checkpoint

id2name = ['objects', 'access_aisle', 'curbside', 'dp_no_aisle', 'dp_one_aisle', 'dp_two_aisle', 'one_aisle', 'two_aisle']

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()


args.dataset_file = 'coco'
args.coco_path = "/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DisabilityParkingCV.v5i.coco" # the path of coco
args.fix_size = False

dataset_test = build_dataset(image_set='test', args=args)   

for i, (image, target) in enumerate(dataset_test):
    img_id = dataset_test.ids[i]
    filename = dataset_test.coco.loadImgs(img_id)[0]['file_name']
    print(filename)
    exit()


for image, targets in dataset_test:
    if len(targets['labels']) > 0: # If there's an object

        # Ground truth
        gt_dict = {
            'boxes': targets['boxes'],
            'image_id': targets['image_id'],
            'size': targets['size'],
            'box_label': [id2name[int(item)] for item in targets['labels']],
        }
        vslzr = COCOVisualizer()
        vslzr.visualize(image, gt_dict, savedir='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/validation_output/gt/')

        # Model output
        output = model.cuda()(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

        threshold = 0.5 # set a threshold
        
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > threshold

        box_label = [id2name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([image.shape[1], image.shape[2]]),
            'image_id': targets['image_id'],
            'box_label': box_label
        }
        vslzr.visualize(image, pred_dict, savedir='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/validation_output/predicted/', dpi=100)

        # print(boxes, labels, scores)