import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from torchvision import transforms
from transformers import get_scheduler
import random
import numpy as np
import shutil
import os
import yaml
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex, Accuracy, Specificity, Recall # 若torchmetrics版本太低就会报错，需要更新到最新
from unet import UNet as UNet # 引入需要的模型
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import logging
import tempfile

logger = get_logger(__name__, log_level="INFO")

# Custom dataset class
# dataset/
# ├── img/
# │   ├── train/
# │   ├── validation/
# │   └── test/
# └── mask/
#     ├── train/
#     ├── validation/
#     └── test/
# enlarge_dataset/
# ├── img/
# │   └── train/
# └── mask/
#     └── train/
# Custom dataset class for loading images and masks from file structure
class CustomHFDataset(Dataset):
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

# Training function
def train(model, dataloader, optimizer, lr_scheduler, accelerator, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle deep supervision outputs
        if isinstance(outputs, tuple):
            main_output = outputs[0]
            ds_output3 = outputs[1]
            ds_output2 = outputs[2]
            
            # Calculate main and auxiliary losses
            main_loss = nn.BCEWithLogitsLoss()(main_output, targets.float())
            aux_loss3 = nn.BCEWithLogitsLoss()(ds_output3, targets.float())
            aux_loss2 = nn.BCEWithLogitsLoss()(ds_output2, targets.float())
            
            # Apply deep supervision with weighted losses
            # Start with more weight on aux losses and gradually decrease
            aux_weight = max(0.15, 0.3 * (1.0 - epoch / (accelerator.state.num_processes * 10)))
            main_weight = 1.0 - 2 * aux_weight
            
            loss = main_weight * main_loss + aux_weight * aux_loss3 + aux_weight * aux_loss2
            
            # For logging purposes
            accelerator.log({
                "train_main_loss": main_loss.item(),
                "train_aux_loss3": aux_loss3.item(),
                "train_aux_loss2": aux_loss2.item(),
                "aux_weight": aux_weight
            }, step=epoch)
        else:
            # For backward compatibility or inference mode
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            main_output = outputs  # For consistent variable naming below
        
        # Backward and optimize
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    # Log average loss for the epoch
    accelerator.log({"train_loss": total_loss / len(dataloader)}, step=epoch)

def validate(model, dataloader, accelerator, metrics, epoch):
    model.eval()
    val_loss = 0
    for metric in metrics.values():
        metric.reset()

    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle deep supervision outputs - use only main output for evaluation
            if isinstance(outputs, tuple):
                main_output = outputs[0]
                ds_output3 = outputs[1]
                ds_output2 = outputs[2]
                
                # Calculate losses for logging
                main_loss = nn.BCEWithLogitsLoss()(main_output, targets.float())
                aux_loss3 = nn.BCEWithLogitsLoss()(ds_output3, targets.float())
                aux_loss2 = nn.BCEWithLogitsLoss()(ds_output2, targets.float())
                
                # Apply same weighting as in training
                aux_weight = max(0.15, 0.3 * (1.0 - epoch / (accelerator.state.num_processes * 10)))
                main_weight = 1.0 - 2 * aux_weight
                
                loss = main_weight * main_loss + aux_weight * aux_loss3 + aux_weight * aux_loss2
                
                # Use main output for metrics
                outputs = main_output
            else:
                # For backward compatibility
                loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            
            val_loss += loss.item()

            # Calculate metrics using main output
            preds = torch.sigmoid(outputs) > 0.5
            targets_int = targets.int()
            metrics["iou"](preds, targets_int)
            metrics["dice"](preds, targets_int)
            metrics["acc"](preds, targets_int)
            metrics["se"](preds, targets_int)
            metrics["sp"](preds, targets_int)

            # Upload images for visualization
            if batch_count == 0 and (epoch + 1) % args.logging_epochs == 0:
                images_cell = inputs[:4].detach().cpu().numpy()
                images_mask = targets[:4].detach().cpu().numpy()
                pred_masks = preds[:4].detach().cpu().numpy()

                images_cell_list = []
                images_mask_list = []
                pred_mask_list = []

                for i in range(4):
                    if images_cell[i].shape[0] == 1:
                        img_cell = Image.fromarray((images_cell[i].squeeze(0) * 255).astype(np.uint8), mode='L')
                    else:
                        img_cell = Image.fromarray((images_cell[i].transpose(1, 2, 0) * 255).astype(np.uint8))

                    img_mask = Image.fromarray((images_mask[i].squeeze(0) * 255).astype(np.uint8), mode='L')
                    pred_mask = Image.fromarray((pred_masks[i].squeeze(0) * 255).astype(np.uint8), mode='L')
                    
                    images_cell_list.append(img_cell)
                    images_mask_list.append(img_mask)
                    pred_mask_list.append(pred_mask)

                accelerator.log({
                    "val_image": [wandb.Image(img_cell) for img_cell in images_cell_list],
                    "val_mask": [wandb.Image(img_mask) for img_mask in images_mask_list],
                    "val_pred": [wandb.Image(pred_mask) for pred_mask in pred_mask_list]
                },
                step=epoch)
            batch_count += 1

    accelerator.log({
        "val_loss": val_loss / len(dataloader),
        "val_iou": metrics["iou"].compute().item(),
        "val_dice": metrics["dice"].compute().item(),
        "val_acc": metrics["acc"].compute().item(),
        "val_se": metrics["se"].compute().item(),
        "val_sp": metrics["sp"].compute().item(),
    },
    step=epoch)

# Test function
def test(model, dataloader, accelerator, metrics):
    model.eval()
    test_loss = 0
    for metric in metrics.values():
        metric.reset()

    total_inference_time = 0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = model(inputs)
            end_time.record()
            
            # Wait for GPU to finish
            torch.cuda.synchronize()
            
            # Calculate inference time in milliseconds
            batch_inference_time = start_time.elapsed_time(end_time)
            total_inference_time += batch_inference_time
            batch_count += 1
            
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            test_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            targets_int = targets.int()  # 转换为整数张量
            metrics["iou"](preds, targets_int)
            metrics["dice"](preds, targets_int)
            metrics["acc"](preds, targets_int)
            metrics["se"](preds, targets_int)
            metrics["sp"](preds, targets_int)

    # Calculate average inference time
    avg_inference_time = total_inference_time / batch_count if batch_count > 0 else 0
    
    # Log metrics including inference time
    metrics_dict = {
        "test_loss": test_loss / len(dataloader),
        "test_iou": metrics["iou"].compute().item(),
        "test_dice": metrics["dice"].compute().item(),
        "test_acc": metrics["acc"].compute().item(),
        "test_se": metrics["se"].compute().item(),
        "test_sp": metrics["sp"].compute().item(),
        "test_avg_inference_time_ms": avg_inference_time,
    }
    
    accelerator.log(metrics_dict)
    
    # Print inference speed for easy comparison
    logger.info(f"Test Inference Speed: {avg_inference_time:.2f} ms/batch (batch size: {dataloader.batch_size})")
    
    return metrics_dict

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Main function
def main(args):
    set_seed(args.seed)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(log_with="wandb", project_config=accelerator_project_config)  # 使用WandB日志记录器
    # 初始化wandb追踪器
    # Initialize wandb tracker with custom run name if provided
    # Extract model architecture name for default run name
    model_class = UNet.__module__
    if '.' in model_class:
        model_arch_name = model_class.split('.')[-1]  # Get the last part of the module path
    else:
        model_arch_name = model_class
    
    # Use provided run_name or fallback to model architecture name
    run_name = args.run_name if args.run_name else model_arch_name
    
    wandb_init_kwargs = {
        "project_name": args.project,
        "config": dict(vars(args)),
        "init_kwargs": {
            "wandb": {
                "name": run_name
            }
        }
    }
    accelerator.init_trackers(**wandb_init_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 设置color_mode
    color_mode = "RGB" if args.color_mode == "rgb" else "L"

    # 设定合法扩充方式
    valid_choices = ["org", "enlarge", "flip_augmentation", "rotate_augmentation"]
    
    # 检查未被允许的扩充方式并进行提醒
    invalid_choices = set(args.enlarge_and_org_choice) - set(valid_choices)
    if invalid_choices:
        raise ValueError(f"Invalid augmentation choices found: {', '.join(invalid_choices)}. Please choose from {', '.join(valid_choices)}.")

    train_dataset_list = []

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # 使用原始数据集
    if "org" in args.enlarge_and_org_choice:
        train_dataset_list.append(
            CustomHFDataset(
                img_dir=os.path.join(args.dataset, 'img/train'),
                mask_dir=os.path.join(args.dataset, 'mask/train'),
                transform=transform,
                mask_transform=mask_transform,
                color_mode=color_mode
            )
        )

    # 使用扩充数据集
    if "enlarge" in args.enlarge_and_org_choice and args.enlarge_train_dataset:
        train_dataset_list.append(
            CustomHFDataset(
                img_dir=os.path.join(args.enlarge_train_dataset, 'img/train'),
                mask_dir=os.path.join(args.enlarge_train_dataset, 'mask/train'),
                transform=transform,
                mask_transform=mask_transform,
                color_mode=color_mode
            )
        )

    # 添加翻转扩充
    if "flip_augmentation" in args.enlarge_and_org_choice:
        transform_with_flip = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
        ])
        mask_transform_with_flip = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
        ])
        train_dataset_list.append(
            CustomHFDataset(
                img_dir=os.path.join(args.dataset, 'img/train'),
                mask_dir=os.path.join(args.dataset, 'mask/train'),
                transform=transform_with_flip,
                mask_transform=mask_transform_with_flip,
                color_mode=color_mode
            )
        )

    # 添加旋转扩充
    if "rotate_augmentation" in args.enlarge_and_org_choice:
        transform_with_rotate = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
        ])
        mask_transform_with_rotate = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
        ])
        train_dataset_list.append(
            CustomHFDataset(
                img_dir=os.path.join(args.dataset, 'img/train'),
                mask_dir=os.path.join(args.dataset, 'mask/train'),
                transform=transform_with_rotate,
                mask_transform=mask_transform_with_rotate,
                color_mode=color_mode
            )
        )

    # 合并所有扩充方式的训练集
    if len(train_dataset_list) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    else:
        train_dataset = train_dataset_list[0]

    val_dataset = CustomHFDataset(
        img_dir=os.path.join(args.dataset, 'img/validation'),
        mask_dir=os.path.join(args.dataset, 'mask/validation'),
        transform=transform,
        mask_transform=mask_transform,
        color_mode=color_mode
    )
    test_dataset = CustomHFDataset(
        img_dir=os.path.join(args.dataset, 'img/test'),
        mask_dir=os.path.join(args.dataset, 'mask/test'),
        transform=transform,
        mask_transform=mask_transform,
        color_mode=color_mode
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, pin_memory=True)

    # Initialize model, optimizer, and scheduler
    in_channels = 3 if args.color_mode == "rgb" else 1  # 根据color_mode设置输入通道数
    model = UNet(in_channels, 1) # in_chanel=in_channel, out_channel=1
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.num_epochs * len(train_dataloader)) # cosine

    # Prepare model and dataloaders with accelerator
    model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    # 指标修复尝试修改
    metrics = {
        "iou": JaccardIndex(task="binary", num_classes=2).to(accelerator.device),  # IOU
        "dice": DiceScore(average="micro", num_classes=2).to(accelerator.device),  # Dice
        "acc": Accuracy(task="binary").to(accelerator.device),  # 准确率
        "se": Recall(task="binary").to(accelerator.device),
        "sp": Specificity(task="binary").to(accelerator.device)
    }

    logger.info(f"Training {args.project} for {args.num_epochs} epochs")
    logger.info(f"Using {args.color_mode} images")
    logger.info(f"Using augmentation methods: {', '.join(args.enlarge_and_org_choice)}")
    logger.info(f"Total training samples: {len(train_dataset)}")
    logger.info(f"Using {len(val_dataset)} validation samples")
    logger.info(f"Using {len(test_dataset)} test samples")

    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            first_epoch = int(path.split("-")[1])
            logger.info(f"Resuming from epoch {first_epoch}")

    # Training loop
    progress_bar = tqdm(range(args.num_epochs), desc="Training", unit="epoch")
    progress_bar.update(first_epoch)
    for epoch in range(first_epoch, args.num_epochs):
        train(model, train_dataloader, optimizer, lr_scheduler, accelerator, epoch)
        validate(model, val_dataloader, accelerator, metrics, epoch)
        progress_bar.update(1)

        if accelerator.is_main_process:
            # 保存模型
            if (epoch + 1) % args.checkpoint_epochs == 0:
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Test the model
        test(model, test_dataloader, accelerator, metrics)

        # Save the model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = accelerator.unwrap_model(model)  # 获取原始模型
        
        # Extract model architecture name from the import module path
        # For example, from "vs_models.fscanet" get "fscanet"
        model_module = model_to_save.__class__.__module__
        if '.' in model_module:
            model_arch_name = model_module.split('.')[-1]  # Get the last part of the module path
        else:
            model_arch_name = model_module
        
        # Create directory with project name if it doesn't exist
        project_dir = os.path.join(args.output_dir, args.project)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        
        # Create filename with model architecture name and augmentation choices
        model_filename = f'{model_arch_name}_{"_".join(["-" + choice + "-" for choice in args.enlarge_and_org_choice])}_epoch{args.num_epochs}.pt'
        model_save_path = os.path.join(project_dir, model_filename)
        
        torch.save(model_to_save.state_dict(), model_save_path)
        logger.info(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UNet model with WandB and Accelerate")
    parser.add_argument('--config', type=str, default="../seg_train_config.yaml", help='config file path, replace the default config, 用于指定配置文件路径，不用费事一个一个参数的指定')
    parser.add_argument("--project", type=str, default="nameless", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for this wandb run (if not specified, will use default naming)")
    parser.add_argument("--logging_dir", type=str, default="./log", help="Directory to log WandB logs")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save WandB logs")
    parser.add_argument("--dataset", type=str, required=None, help="Dataset directory path")
    parser.add_argument("--enlarge_train_dataset", type=str, default=None, help="使用的扩充数据集的路径")
    parser.add_argument(
        "--enlarge_and_org_choice", 
        type=str, 
        nargs='+', 
        default=["org"], 
        # choices=["org", "enlarge", "flip_augmentation", "rotate_augmentation"], 在上面扩充方式列表中进行检查
        help="扩充方式列表， 例如： org enlarge flip_augmentation rotate_augmentation"
    )

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--seed", type=int, default=7777777, help="Random seed")
    parser.add_argument("--color_mode", type=str, default="rgb", choices=["rgb", "grayscale"], help="Color mode of the images")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training, latest checkpoint will be used if value is 'latest'")
    parser.add_argument("--checkpoint_epochs", type=int, default=5, help="Epoch to resume training from")
    parser.add_argument("--checkpoints_total_limit" , type=int, default=5, help="Total number of checkpoints to keep")
    parser.add_argument("--logging_epochs", type=int, default=1, help="Log metrics every n epochs")
    parser.add_argument("--num_workers", type=int, default=0, 
                    help="Number of worker processes for data loading (0 for main process only)")

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 用config字典中的参数替换args
            for key, value in config.items():
                setattr(args, key, value)

    if args.dataset is None:
        raise ValueError("Please provide a dataset directory path")

    # 确保配置文件的enlarge_and_org_choice是列表
    if isinstance(args.enlarge_and_org_choice, str):
        args.enlarge_and_org_choice = args.enlarge_and_org_choice.split()
    
    main(args)
