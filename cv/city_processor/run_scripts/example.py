import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the Disability Parking Detector and the Locator/Characterizer models
from DisabilityParkingIdentifier import DisabilityParkingIdentifier
from locator_models.CoDETR_SWIN_LocatorModel import DisabilityParkingSpaceLocatorCODetrSWIN
from locator_models.YOLO_OBB_SegmenterModel import DisabilityParkingSpaceSegmenterYOLO

# Load the config and checkpoint for CoDETR
codetr_config_file = 'locator_models/configs/co_dino_swin_disabilityparking_aug_batchsize2.py'
codetr_checkpoint_file = 'locator_models/checkpoints/epoch_23.pth'
codetr_detection_threshold = 0.2 # confidence

# Load the checkpoint for YOLO
yolo_checkpoint_file = 'locator_models/checkpoints/best.pt'
yolo_detection_threshold = 0.3 # confidence

# Initialize the model objects
locator_model = DisabilityParkingSpaceLocatorCODetrSWIN(codetr_config_file, codetr_checkpoint_file, confidence_threshold=codetr_detection_threshold, imgsz=512)
obb_model = DisabilityParkingSpaceSegmenterYOLO(yolo_checkpoint_file, confidence_threshold=yolo_detection_threshold)
characterizer_models = [obb_model]

# Point to the source orthorectified tiles
project_name = 'example' # a unique project name
source_tile_dir = 'run_scripts/example_data/small_example_region/tiles/static/king/256_20' # source tiles
output_dir = 'run_scripts/example_data/output' # output dir
bbox = (('47.6209184', '-122.3465409'), ('47.6197961', '-122.3445139')) # latlong bounding box to detect within
visualize_dir = 'run_scripts/example_data/output/viz' # optional dir to save cropped parking space visualizations to

# Initialize detector 
locator = DisabilityParkingIdentifier(locator_model, characterizer_models)

    # From scratch...
locator.initialize(project_name,
                   source_tile_dir, # source tiles
                   output_dir, # output dir
                   bbox[0], # bbox top left
                   bbox[1], # bbox bottom right
)

    # Or resuming from existing run
# locator.initialize_resume('/gscratch/makelab/jaredhwa/DisabilityParkingCR/cv/city_processor/run_scripts/temp_output/teaserfig/progress_file.json',
#                    bbox[0], # bbox top left
#                    bbox[1], # bbox bottom right
# )

# Run. Output will be saved to output_dir 
locator.run(visualize_dir=visualize_dir)

