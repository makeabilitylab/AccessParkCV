from DisabilityParkingIdentifier import DisabilityParkingIdentifier
from CoDETR_SWIN_LocatorModel import DisabilityParkingSpaceLocatorCODetrSWIN
from YOLO_OBB_SegmenterModel import DisabilityParkingSpaceSegmenterYOLO

codetr_config_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/co_dino_swin_disabilityparking_aug_batchsize2.py'
codetr_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/work_dirs/co_dino_swin_disabilityparking_aug_batchsize2/epoch_23.pth'
codetr_detection_threshold = 0.2 # confidence

yolo_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb/train2/weights/best.pt'
yolo_detection_threshold = 0.3 # confidence

locator_model = DisabilityParkingSpaceLocatorCODetrSWIN(codetr_config_file, codetr_checkpoint_file, confidence_threshold=codetr_detection_threshold, imgsz=512)
obb_model = DisabilityParkingSpaceSegmenterYOLO(yolo_checkpoint_file, confidence_threshold=yolo_detection_threshold)
characterizer_models = [obb_model]

# LOCATIONS

# Seattle Choropleth
project_name = 'seattle_choropleth'
source_tile_dir = '/gscratch/scrubbed/jaredhwa/DisabilityParking/data/tile2net/seattle/tiles/static/king/256_20'
output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_choropleth/predicted'
bbox = (('47.69889', '-122.43837'), ('47.60158', '-122.23933'))
visualize_dir = None


locator = DisabilityParkingIdentifier(locator_model, characterizer_models)
locator.initialize(project_name,
                   source_tile_dir, # source tiles
                   output_dir, # output dir
                   bbox[0], # bbox top left
                   bbox[1], # bbox bottom right
)

locator.run(visualize_dir=None)

