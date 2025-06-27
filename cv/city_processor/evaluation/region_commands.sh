#!/bin/bash

# # Seattle Northgate
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/ground_truth/seattle_northgate_groundtruth/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/predicted/seattle_northgate_predicted/total_spaces.json \
#                                 --pair-names seattle_northgate \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/predicted/YOLO_segmented_detections \
#                                 --no-visuals


# # Seattle Roosevelt
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/ground_truth/seattle_roosevelt_groundtruth/total_spaces.json  /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/predicted/seattle_roosevelt/total_spaces.json \
#                                 --pair-names seattle_roosevelt \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/predicted/YOLO_segmented_detections \
#                                 --no-visuals


# # DC Audifield
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/ground_truth/dc_audifield_groundtruth/total_spaces.json  /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/predicted/dc_audifield_predicted/total_spaces.json \
#                                 --pair-names dc_audifield \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/predicted/YOLO_segmented_detections \
#                                 --no-visuals


# # DC USPS
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/ground_truth/dc_usps_groundtruth/total_spaces.json  /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/predicted/dc_usps_groundtruth/total_spaces.json \
#                                 --pair-names dc_usps \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/predicted/YOLO_segmented_detections \
#                                 --no-visuals


# # Waltham
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/ground_truth/mass_waltham/total_spaces.json  /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/predicted/mass_waltham/total_spaces.json \
#                                 --pair-names mass_waltham \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/predicted/YOLO_segmented_detections \
#                                 --no-visuals

# # # LA Torrance
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/ground_truth/la_torrance/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/predicted/la_torrance/total_spaces.json \
#                                 --pair-names la_torrance \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/predicted/YOLO_segmented_detections \
#                                 --no-visuals

# # LA ktown
# python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/ground_truth/la_ktown/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/predicted/la_ktown/total_spaces.json \
#                                 --pair-names la_ktown \
#                                 --iou 0.3 \
#                                 --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate \
#                                 --gt-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/ground_truth/groundtruth_segmented_detections \
#                                 --pred-images /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/predicted/YOLO_segmented_detections \
#                                 --no-visuals


# # Aggregates

# In training
python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/ground_truth/dc_audifield_groundtruth/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_audifield/predicted/dc_audifield_predicted/total_spaces.json \
                                --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/ground_truth/dc_usps_groundtruth/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/dc_usps/predicted/dc_usps_groundtruth/total_spaces.json \
                                --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/ground_truth/seattle_northgate_groundtruth/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_northgate/predicted/seattle_northgate_predicted/total_spaces.json  \
                                --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/ground_truth/seattle_roosevelt_groundtruth/total_spaces.json  /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/seattle_roosevelt/predicted/seattle_roosevelt/total_spaces.json \
                                --pair-names DCAudifield DCUSPS SeattleNorthgate SeattleRoosevelt \
                                --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate_in_training_set \
                                --iou 0.3 \
                                --no-visuals 

# Outside training
python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/ground_truth/la_ktown/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/predicted/la_ktown/total_spaces.json \
                                --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/ground_truth/la_torrance/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/predicted/la_torrance/total_spaces.json \
                                --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/ground_truth/mass_waltham/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/mass_waltham/predicted/mass_waltham/total_spaces.json \
                                --pair-names LAKtown LATorrance MassWaltham \
                                --iou 0.3 \
                                --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate_outside_training_set \
                                --no-visuals

# Just LA
python multi_pair_evaluation.py --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/ground_truth/la_ktown/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_ktown/predicted/la_ktown/total_spaces.json \
                                --pair /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/ground_truth/la_torrance/total_spaces.json /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/regions/la_torrance/predicted/la_torrance/total_spaces.json \
                                --pair-name LAKtown LATorrance \
                                --iou 0.3 \
                                --output /gscratch/makelab/jaredhwa/DisabilityParking/cv/city_processor/evaluation/aggregate_la \
                                --no-visuals
