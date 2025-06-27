from DisabilityParkingIdentifier import DisabilityParkingIdentifier
from LocatorFromGroundTruthModel import DisabilityParkingGroundTruthLocator
from YOLO_OBB_SegmenterModel import DisabilityParkingSpaceSegmenterYOLO

topleft_coords_override=None

# Seattle Roosevelt
# project_name = 'seattle_roosevelt_groundtruth'
# label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/labels'
# source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/seattle_roosevelt/tiles/static/king/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/ground_truth'
# bbox_top_left = ('47.681744', '-122.325996')
# bbox_bottom_right = ('47.67550974029736', '-122.31689651540462')
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/ground_truth/groundtruth_segmented_detections'

# Seattle Northgate
# project_name = 'seattle_northgate_groundtruth'
# label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/labels'
# source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/seattle_northgate/tiles/static/king/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/ground_truth'
# bbox_top_left = ('47.710431', '-122.328963')
# bbox_bottom_right = ('47.704166826506054', '-122.31964309743587')
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/ground_truth/groundtruth_segmented_detections'

# DC Audifield
# project_name = 'dc_audifield_groundtruth'
# label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/labels'
# source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/dc_audifield/tiles/static/dc/512_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/ground_truth'
# bbox_top_left = ('38.8720477', '-77.0170474')
# bbox_bottom_right = ('38.86564216992963', '-77.0090103149414')
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/ground_truth/groundtruth_segmented_detections'
# topleft_coords_override=(299960, 401224) 

# DC USPS
# project_name = 'dc_usps_groundtruth'
# label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/labels'
# source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/dc_usps/tiles/static/dc/512_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/ground_truth'
# bbox_top_left = ('38.921611', '-76.995726')
# bbox_bottom_right = ('38.91534589480659', '-76.98772430419922')
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/ground_truth/groundtruth_segmented_detections'

# # Mass
# project_name = 'mass_waltham'
# label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/labels'
# source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/mass_waltham/tiles/static/ma/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/ground_truth'
# bbox_top_left = ('42.372182', '-71.222412')
# bbox_bottom_right = ('42.36615433532088', '-71.21440887451172')
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/ground_truth/groundtruth_segmented_detections'

# LA
# project_name = 'la_ktown'
# label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/labels'
# source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/la_ktown/tiles/static/la/256_20'
# output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/ground_truth'
# bbox_top_left = ('34.064376', '-118.305435')
# bbox_bottom_right = ('34.057779382846725', '-118.29734802246094')
# visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/ground_truth/groundtruth_segmented_detections'

project_name = 'la_torrance'
label_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/labels'
source_tile_path = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/region_validator/validated_regions/la_torrance/tiles/static/la/256_20'
output_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/ground_truth'
bbox_top_left = ('33.835593', '-118.356391')
bbox_bottom_right = ('33.828786509874305', '-118.34815979003906')
visualize_dir = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/ground_truth/groundtruth_segmented_detections'



locator_model = DisabilityParkingGroundTruthLocator(label_path)
characterizer_models = [locator_model]
# yolo_checkpoint_file = '/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runsobb/train/weights/best.pt'
# yolo_detection_threshold = 0.2 # confidence
# obb_model = DisabilityParkingSpaceSegmenterYOLO(yolo_checkpoint_file, confidence_threshold=yolo_detection_threshold)
# characterizer_models = [obb_model]


test = DisabilityParkingIdentifier(locator_model, characterizer_models)
test.initialize(project_name,
                source_tile_path, # source tiles
                output_dir, # output dir
                bbox_top_left, # bbox top left
                bbox_bottom_right, # bbox bottom right
)

test.run(visualize_dir=visualize_dir, topleft_coords_override=topleft_coords_override)
# test.run(visualize_dir='/gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/ground_truth/YOLO_segmented_detections_actualmodel')
# test.run()