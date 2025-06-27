import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import Trainer, TrainingArguments


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# Custom dataset for the parking spaces
class ParkingSpaceDataset(Dataset):
    def __init__(self, img_dir, annotation_file, processor, is_train=True):
        self.img_dir = img_dir
        self.processor = processor
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build category ID to label mapping
        self.cat_id_to_label = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Build image ID to filename mapping
        self.img_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image ID
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Get the list of image IDs (only those with annotations)
        self.img_ids = list(self.annotations.keys())
        
        # Data augmentation for training
        # if is_train:
        #     self.transform = T.Compose([
        #         T.RandomHorizontalFlip(p=0.5),
        #         T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         T.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
        #     ])
        # else:
        #     self.transform = None
        self.transform = None

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        filename = self.img_id_to_file[img_id]
        image_path = os.path.join(self.img_dir, filename)
        
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Get annotations for this image
        img_anns = self.annotations[img_id]
        
        # Prepare target format required by DINO
        boxes = []
        classes = []
        area = []
        iscrowd = []
        
        for ann in img_anns:
            # COCO format has [x_min, y_min, width, height]
            # Convert to [x_min, y_min, x_max, y_max] for DETR format
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            
            # Get class id (subtract 1 as DETR/DINO classes are 0-indexed)
            # Note: if class_id 0 is background in your case, don't subtract 1
            class_id = ann["category_id"]
            classes.append(class_id)
            
            # Include area and iscrowd info
            area.append(ann["area"])
            iscrowd.append(ann["iscrowd"])
        
        # Convert to tensors
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(classes, dtype=torch.long)
        target["area"] = torch.tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.long)
        
        # Process inputs through the processor
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target_prepared = encoding["labels"][0]  # Take first element as we're processing single images
        
        return {
            "pixel_values": pixel_values,
            "labels": target_prepared
        }


# Function to collate batches
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    
    batch = {
        "pixel_values": pixel_values,
        "labels": labels
    }
    return batch


# Visualization function for predictions
def visualize_predictions(image, predictions, processor, cat_id_to_label, threshold=0.5):
    # Get image shape
    image_size = image.size[::-1]  # (W, H) -> (H, W)
    
    # Get predictions
    boxes = predictions["pred_boxes"].cpu().detach().numpy()
    scores = predictions["scores"].cpu().detach().numpy()
    labels = predictions["labels"].cpu().detach().numpy()
    
    # Filter by threshold
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)
        
    # Create matplotlib figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw all bounding boxes
    for box, score, label in zip(boxes, scores, labels):
        # Convert normalized boxes to pixel coordinates (if needed)
        box = box * np.array([image_size[1], image_size[0], image_size[1], image_size[0]])
        
        # Draw the bounding box
        x1, y1, x2, y2 = box.astype(int)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
        
        # Add label and score
        class_name = cat_id_to_label[label]
        text = f"{class_name}: {score:.2f}"
        plt.gca().text(x1, y1, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Main training function
def train_dino_model(
    train_dir, 
    train_annotations, 
    val_dir, 
    val_annotations, 
    output_dir="dino_parking_model", 
    model_name="facebook/dino-vits16",
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-5
):
    """
    Train a DINO model on the parking space dataset.
    
    Args:
        train_dir: Directory containing training images
        train_annotations: Path to training annotations file
        val_dir: Directory containing validation images
        val_annotations: Path to validation annotations file
        output_dir: Directory to save model checkpoints
        model_name: Name of the pretrained model to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DINO model and processor
    print(f"Loading pretrained model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        num_labels=8,  # We have 8 classes in our dataset
        ignore_mismatched_sizes=True
    )
    
    # Load category mapping from annotations
    with open(train_annotations, 'r') as f:
        coco_data = json.load(f)
    
    # Create category ID to label mapping
    cat_id_to_label = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Update the model's id2label and label2id mappings
    model.config.id2label = cat_id_to_label
    model.config.label2id = {v: k for k, v in cat_id_to_label.items()}
    
    # Create datasets
    train_dataset = ParkingSpaceDataset(train_dir, train_annotations, processor, is_train=True)
    val_dataset = ParkingSpaceDataset(val_dir, val_annotations, processor, is_train=False)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        report_to="none",  # Disable reporting to avoid wandb etc. requirements
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    model.save_pretrained(f"{output_dir}/final_model")
    processor.save_pretrained(f"{output_dir}/final_model")
    
    return model, processor, cat_id_to_label


# Inference function
def run_inference(model, processor, cat_id_to_label, image_path, threshold=0.5):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert outputs to predictions
    predictions = processor.post_process_object_detection(
        outputs, 
        threshold=threshold,
        target_sizes=[(image.height, image.width)]
    )[0]
    
    # Visualize
    visualize_predictions(image, predictions, processor, cat_id_to_label, threshold)
    
    return predictions


# Parse command line arguments and run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINO model for parking space detection")
    
    # Dataset paths
    parser.add_argument("--train-dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--train-annotations", type=str, required=True, help="Path to training annotations file")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory containing validation images")
    parser.add_argument("--val-annotations", type=str, required=True, help="Path to validation annotations file")
    parser.add_argument("--test-image", type=str, help="Path to a test image for inference after training", default=None)
    
    # Model parameters
    parser.add_argument("--output-dir", type=str, default="dino_parking_model", help="Output directory for model checkpoints")
    parser.add_argument("--model-name", type=str, default="facebook/dino-vits16", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for optimizer")
    
    # Inference only mode
    parser.add_argument("--inference-only", action="store_true", help="Run inference only on the test image using a saved model")
    parser.add_argument("--saved-model", type=str, help="Path to a saved model for inference", default=None)
    
    args = parser.parse_args()
    
    if args.inference_only:
        if not args.saved_model:
            print("Error: --saved-model must be specified when using --inference-only")
            exit(1)
        if not args.test_image:
            print("Error: --test-image must be specified when using --inference-only")
            exit(1)
            
        # Load saved model for inference
        processor = AutoImageProcessor.from_pretrained(args.saved_model)
        model = AutoModelForObjectDetection.from_pretrained(args.saved_model)
        
        # Load category mapping from annotations
        with open(args.val_annotations, 'r') as f:
            coco_data = json.load(f)
        cat_id_to_label = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Run inference
        predictions = run_inference(model, processor, cat_id_to_label, args.test_image)
        
        # Print predictions
        for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {cat_id_to_label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
    else:
        # Train the model
        model, processor, cat_id_to_label = train_dino_model(
            train_dir=args.train_dir,
            train_annotations=args.train_annotations,
            val_dir=args.val_dir,
            val_annotations=args.val_annotations,
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Run inference on a test image if provided
        if args.test_image:
            predictions = run_inference(model, processor, cat_id_to_label, args.test_image)
            
            # Print predictions
            for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {cat_id_to_label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )