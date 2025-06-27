import pandas as pd

# Load the training log
# results = pd.read_csv('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/segmenter/YOLO/runs/larger_dataset/obb/train2/results.csv')
# results = pd.read_csv('/gscratch/makelab/jaredhwa/DisabilityParking/cv_study/models/disabilityparking_identifier/runs/detect/no_loading_zones/train/results.csv')
results = pd.read_csv('/gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/YOLO/runs/largerdataset/detect/train/results.csv')


# Strip spaces
results.columns = results.columns.str.strip()

# Calculate fitness
results["fitness"] = results["metrics/mAP50(B)"] * 0.1 + results["metrics/mAP50-95(B)"] * 0.9
print(list(enumerate(results["fitness"])))

# Find the epoch with the highest fitness
best_epoch = results['fitness'].idxmax() + 1

print(f"Best model was saved at epoch: {best_epoch}")