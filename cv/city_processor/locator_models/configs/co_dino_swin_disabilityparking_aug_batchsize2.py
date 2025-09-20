# The new config inherits a base config to highlight the necessary modification
_base_ = 'co_dino_5scale_swin_large_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=[dict(
#         bbox_head=dict(num_classes=7),
#         mask_head=dict(num_classes=7))])

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('access_aisle', 'curbside','dp_no_aisle','dp_one_aisle','dp_two_aisle','one_aisle','two_aisle')
data = dict(
    train=dict(
        img_prefix='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/train_no_aa/',
        classes=classes,
        ann_file='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/train_no_aa/_annotations.coco.json'),
    val=dict(
        img_prefix='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/valid_no_aa/',
        classes=classes,
        ann_file='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/valid_no_aa/_annotations.coco.json'),
    test=dict(
        img_prefix='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/test_no_aa/',
        classes=classes,
        ann_file='/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/dataset/DisabilityParkingCV.v6i.coco-segmentation/test_no_aa/_annotations.coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/tools/projects/configs/co_dino_swin_disabilityparking/co_dino_5scale_swin_large_1x_coco.pth'