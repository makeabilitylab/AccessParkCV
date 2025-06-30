import os
import json
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from tqdm import tqdm

# # Specify the path to model config and checkpoint file
# config_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/co_dino_swin_disabilityparking.py'
# checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/work_dirs/co_dino_swin_disabilityparking/epoch_30.pth'
# # output path
# output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/results/predictions_30epoch_testset.json'
# Specify the path to model config and checkpoint file


# Larger dataset
config_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/co_dino_swin_disabilityparking_aug_batchsize2.py'
checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/work_dirs/co_dino_swin_disabilityparking_aug_batchsize2/epoch_23.pth'
# output path
output_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/results_large_dataset_batchsize2/2predictions_23epoch_testset.json'
process_images_in_folder_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/test_no_aa'
detection_threshold = 0.3 # confidence



# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


def predict_with_model(img_path):
    img_name = os.path.basename(img_path)

    result = inference_detector(model, img_path)

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
                {'category_id': predicted_class,
                'bbox': str((left, top, right, bottom)),
                'score': str(bboxes[detected_obj_idx][4])})

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

process_images_in_folder(process_images_in_folder_dir)