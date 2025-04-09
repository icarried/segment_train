import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import yaml
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex, Accuracy, Specificity, Recall
from unet import UNet as UNet
from tqdm import tqdm
from PIL import Image
import time

# Custom dataset class (copied from your training script)
class CustomHFDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, color_mode="RGB"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.color_mode = color_mode
        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = self.mask_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(img_path).convert(self.color_mode)
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {'image': image, 'label': mask}

def test_model(model, dataloader, device, metrics, save_images=False, save_dir=None):
    """
    Test the model on the given dataloader and return metrics.
    """
    model.eval()
    test_loss = 0
    
    # Reset all metrics
    for metric in metrics.values():
        metric.reset()

    # Track inference time
    total_inference_time = 0
    batch_count = 0
    total_samples = 0
    
    # Save predictions if requested
    if save_images and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pred_dir = os.path.join(save_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs, targets = batch['image'].to(device), batch['label'].to(device)
            batch_size = inputs.shape[0]
            total_samples += batch_size
            
            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle deep supervision outputs if present
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use only main output for evaluation
            
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_inference_time += inference_time
            batch_count += 1
            
            # Calculate loss
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            test_loss += loss.item() * batch_size
            
            # Get predictions
            preds = torch.sigmoid(outputs) > 0.5
            targets_int = targets.int()
            
            # Update metrics
            metrics["iou"](preds, targets_int)
            metrics["dice"](preds, targets_int)
            metrics["acc"](preds, targets_int)
            metrics["se"](preds, targets_int)
            metrics["sp"](preds, targets_int)
            
            # Save predictions if requested
            if save_images and save_dir:
                for i in range(batch_size):
                    # Save prediction
                    pred_img = (preds[i].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    pred_img = Image.fromarray(pred_img, mode='L')
                    pred_img.save(os.path.join(pred_dir, f"pred_{batch_count}_{i}.png"))

    # Calculate average metrics
    results = {
        "test_loss": test_loss / total_samples,
        "test_iou": metrics["iou"].compute().item(),
        "test_dice": metrics["dice"].compute().item(),
        "test_acc": metrics["acc"].compute().item(),
        "test_se": metrics["se"].compute().item(),
        "test_sp": metrics["sp"].compute().item(),
        "test_avg_inference_time_ms": total_inference_time / batch_count,
        "frames_per_second": batch_size * batch_count / (total_inference_time / 1000)
    }
    
    return results

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set color mode
    color_mode = "RGB" if args.color_mode == "rgb" else "L"
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    # Create test dataset
    test_dataset = CustomHFDataset(
        img_dir=os.path.join(args.dataset, 'img/test'),
        mask_dir=os.path.join(args.dataset, 'mask/test'),
        transform=transform,
        mask_transform=mask_transform,
        color_mode=color_mode
    )
    
    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Testing on {len(test_dataset)} samples")
    
    # Initialize model
    in_channels = 3 if args.color_mode == "rgb" else 1
    model = UNet(in_channels, 1) # in_chanel=in_channel, out_channel=1
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Initialize metrics
    metrics = {
        "iou": JaccardIndex(task="binary", num_classes=2).to(device),
        "dice": DiceScore(average="micro", num_classes=2).to(device),
        "acc": Accuracy(task="binary").to(device),
        "se": Recall(task="binary").to(device),
        "sp": Specificity(task="binary").to(device)
    }
    
    # Test the model
    results = test_model(
        model, 
        test_dataloader, 
        device, 
        metrics,
        save_images=args.save_predictions,
        save_dir=args.output_dir
    )
    
    # Print results
    print("\n===== Test Results =====")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test IoU: {results['test_iou']:.4f}")
    print(f"Test Dice: {results['test_dice']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Test Sensitivity: {results['test_se']:.4f}")
    print(f"Test Specificity: {results['test_sp']:.4f}")
    print(f"Avg Inference Time: {results['test_avg_inference_time_ms']:.2f} ms/batch")
    print(f"Processing Speed: {results['frames_per_second']:.2f} frames/second")
    
    # Save results to file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            f.write("===== Test Results =====\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained UNet segmentation model")
    parser.add_argument('--config', type=str, default="../seg_test_config.yaml", help='Config file path')
    parser.add_argument("--model_path", type=str, required=False, help="Path to trained model weights")
    parser.add_argument("--dataset", type=str, required=False, help="Dataset directory path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--color_mode", type=str, default="rgb", choices=["rgb", "grayscale"], help="Color mode of the images")
    parser.add_argument("--output_dir", type=str, default="./test_output", help="Directory to save test results")
    parser.add_argument("--save_predictions", type=bool, default=False, help="Save predicted masks to disk")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # Replace args with config values
            for key, value in config.items():
                setattr(args, key, value)
    
    # Check required arguments
    if args.model_path is None:
        raise ValueError("Please provide a model path with --model_path")
    if args.dataset is None:
        raise ValueError("Please provide a dataset directory path with --dataset")
    
    main(args)