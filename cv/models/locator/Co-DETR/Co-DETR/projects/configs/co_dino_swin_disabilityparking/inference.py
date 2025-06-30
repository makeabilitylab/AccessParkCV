from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np

# Specify the path to model config and checkpoint file
config_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/co_dino_swin_disabilityparking.py'
checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/work_dirs/co_dino_swin_disabilityparking/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
img = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/DisabilityParkingCV.v5i.coco/valid/420_723_498843_png.rf.9193c7e2493c463a6dae17b91297f71c.jpg'


# sdhttps://vinleonardo.com/detecting-objects-in-pictures-and-extracting-their-data-using-mmdetection/
result = inference_detector(model, img)

detection_threshold = 0.3 # confidence
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

print(bboxes.shape)
print(labels_impt)
print(labels_impt_list)
print(labels_class)

print(bboxes)

print(bboxes[labels_impt])
# print(bboxes[labels_impt][0][0])
# print(bboxes[labels_impt][0][1])
# print(bboxes[labels_impt][0][2])
# print(bboxes[labels_impt][0][3])
# print(result)

from PIL import Image

im = Image.open(img)

for detected_obj_idx in labels_impt:
    left = bboxes[detected_obj_idx][0]
    top = bboxes[detected_obj_idx][1]
    right = bboxes[detected_obj_idx][2]
    bottom = bboxes[detected_obj_idx][3]
    im1 = im.crop((left, top, right, bottom))
    im1.save(f'/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/results/test_{detected_obj_idx}.jpg')

# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
# model.show_result(img, result, out_file='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/results/test.jpg')
