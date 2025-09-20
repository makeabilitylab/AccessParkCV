import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from DisabilityParkingIdentifier import DisabilityParkingIdentifier
from locator_models.CoDETR_SWIN_LocatorModel import DisabilityParkingSpaceLocatorCODetrSWIN
from locator_models.YOLO_OBB_SegmenterModel import DisabilityParkingSpaceSegmenterYOLO

# codetr_config_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/co_dino_swin_disabilityparking_augmented.py'
# codetr_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/work_dirs/co_dino_swin_disabilityparking_aug_batchsize4/epoch_25.pth'
# codetr_detection_threshold = 0.2 # confidence

# yolo_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runsobb/train/weights/best.pt'
# yolo_detection_threshold = 0.3 # confidence

codetr_config_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/projects/configs/co_dino_swin_disabilityparking/co_dino_swin_disabilityparking_aug_batchsize2.py'
codetr_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/Co-DETR/Co-DETR/work_dirs/co_dino_swin_disabilityparking_aug_batchsize2/epoch_23.pth'
codetr_detection_threshold = 0.2 # confidence

yolo_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb/train2/weights/best.pt'
yolo_detection_threshold = 0.3 # confidence

locator_model = DisabilityParkingSpaceLocatorCODetrSWIN(codetr_config_file, codetr_checkpoint_file, confidence_threshold=codetr_detection_threshold, imgsz=512)
obb_model = DisabilityParkingSpaceSegmenterYOLO(yolo_checkpoint_file, confidence_threshold=yolo_detection_threshold)
characterizer_models = [obb_model]

# exit()

# LOCATIONS

topleft_coords_override=None

# Seattle Roosevelt
# project_name = 'seattle_roosevelt'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/seattle_roosevelt/tiles/static/king/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/predicted'
# bbox = (('47.681744', '-122.325996'), ('47.67550974029736', '-122.31689651540462'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/predicted/YOLO_segmented_detections'

# Seattle Northgate
# project_name = 'seattle_northgate_predicted'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/seattle_northgate/tiles/static/king/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/predicted'
# bbox = (('47.710431', '-122.328963'), ('47.704166826506054', '-122.31964309743587'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/predicted/YOLO_segmented_detections'

# DC Audifield
# project_name = 'dc_audifield_predicted'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/dc_audifield/tiles/static/dc/512_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/predicted'
# bbox = (('38.8720477', '-77.0170474'), ('38.86564216992963', '-77.0090103149414'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/predicted/YOLO_segmented_detections'
# topleft_coords_override=(299960, 401224) # needed if GT labels topleft are even. 

# DC USPS
# project_name = 'dc_usps_groundtruth'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/dc_usps/tiles/static/dc/512_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/predicted'
# bbox = (('38.921611', '-76.995726'), ('38.91534589480659', '-76.98772430419922'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/predicted/YOLO_segmented_detections'

# Mass Waltham
# project_name = 'mass_waltham'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/mass_waltham/tiles/static/ma/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/predicted'
# bbox = (('42.372182', '-71.222412'), ('42.36615433532088', '-71.21440887451172'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/predicted/YOLO_segmented_detections'

# LA ktown
# project_name = 'la_ktown'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/la_ktown/tiles/static/la/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/predicted'
# bbox = (('34.064376', '-118.305435'), ('34.057779382846725', '-118.29734802246094'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/predicted/YOLO_segmented_detections'

# LA Torrance
# project_name = 'la_torrance'
# source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/la_torrance/tiles/static/la/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/predicted'
# bbox = (('33.835593', '-118.356391'), ('33.828786509874305', '-118.34815979003906'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/predicted/YOLO_segmented_detections'

# Teaserfig
project_name = 'teaserfig'
source_tile_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/seattle_teaserfig/tiles/static/king/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/teaserfig/predicted'
output_dir = '/gscratch/makelab/jaredhwa/DisabilityParkingCR/cv/city_processor/run_scripts/temp_output'
bbox = (('47.6209184', '-122.3465409'), ('47.6197961', '-122.3445139'))
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/teaserfig/predicted/YOLO_segmented_detections'
visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParkingCR/cv/city_processor/run_scripts/temp'


locator = DisabilityParkingIdentifier(locator_model, characterizer_models)
locator.initialize(project_name,
                   source_tile_dir, # source tiles
                   output_dir, # output dir
                   bbox[0], # bbox top left
                   bbox[1], # bbox bottom right
)

# Resuming from existing run
# locator.initialize_resume('/gscratch/makelab/jaredhwa/DisabilityParkingCR/cv/city_processor/run_scripts/temp_output/teaserfig/progress_file.json',
#                    bbox[0], # bbox top left
#                    bbox[1], # bbox bottom right
# )

locator.run(visualize_dir=visualize_dir, topleft_coords_override=topleft_coords_override)

