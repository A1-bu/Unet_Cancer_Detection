!pip install wldhx.yadisk-direct

!curl -L $(yadisk-direct https://disk.yandex.ru/d/-4g-l44mqslQXg) -o data.zip

!unzip data.zip >> /dev/null

import albumentations as A
from albumentations.pytorch import ToTensorV2
# Training transform with various augmentations
train_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
])

# Validation (and test) transform without random augmentation
val_transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiClassBUSIDataset(Dataset):
    """
    Multi-class version of BUSI Dataset:
    - Class 0: Background
    - Class 1: Benign
    - Class 2: Malignant

    Expects a directory structure:
    root_dir/
        benign/
            benign_1.png
            benign_1_mask.png
            ...
        malignant/
            malignant_1.png
            malignant_1_mask.png
            ...
        normal/
            normal_1.png
            normal_1_mask.png
            ...
    """
    def __init__(self, root_dir, image_size=(256, 256), transform=None, samples=None,
                 include_normal=True):
        """
        :param image_size: desired HxW for resizing
        :param transform: any additional torchvision transform
        :param samples: optional list of (image_path, mask_path, class_id) tuples. If None, scans automatically.
        :param include_normal: whether to include normal cases (no lesions)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.include_normal = include_normal

        # Class mapping
        self.class_to_idx = {
            'background': 0,
            'benign': 1,
            'malignant': 2
        }

        if samples is not None:
            # If we already have a subset for train/val
            self.samples = samples
        else:
            self.samples = self._scan_dataset()

    def _scan_dataset(self):
        samples = []

        # Determine which subfolders to scan
        if self.include_normal:
            subfolders = ["benign", "malignant", "normal"]
        else:
            subfolders = ["benign", "malignant"]

        for sf in subfolders:
            folder_path = os.path.join(self.root_dir, sf)
            # find all *mask.png
            mask_paths = glob.glob(os.path.join(folder_path, "*_mask.png"))
            for mp in mask_paths:
                # corresponding image path is mp.replace("_mask", "")
                img_path = mp.replace("_mask", "")
                if os.path.exists(img_path):
                    # Determine class from folder name
                    if sf == "normal":
                        class_id = 0  # Normal cases have no lesions, just background
                    elif sf == "benign":
                        class_id = 1  # Benign
                    elif sf == "malignant":
                        class_id = 2  # Malignant

                    samples.append((img_path, mp, class_id))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, class_id = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0

        binary_mask = (mask > 127).astype(np.uint8)

        if class_id == 0:  # Normal case - all background
            multi_class_mask = np.zeros_like(binary_mask)
        else:  # Benign or Malignant
            multi_class_mask = binary_mask * class_id

        if self.transform is not None:
            # Here you would apply any data augmentation
            pass

        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]

        mask_tensor = torch.from_numpy(multi_class_mask).long()  # [H, W]

        return image_tensor, mask_tensor

def create_train_val_datasets(root_dir, val_ratio=0.2, image_size=(256, 256), include_normal=True, stratify=True):
    """
    Splits the dataset into train and validation sets with optional stratification by class.
    Returns two MultiClassBUSIDataset objects.

    Args:
        root_dir: Base directory containing the dataset folders
        val_ratio: Percentage of data to use for validation
        image_size: Tuple of (height, width) for resizing
        include_normal: Whether to include normal cases
        stratify: Whether to maintain class balance in train/val splits

    Returns:
        train_dataset, val_dataset
    """
    import os
    import glob
    import random
    from collections import defaultdict

    if include_normal:
        subfolders = ["benign", "malignant", "normal"]
    else:
        subfolders = ["benign", "malignant"]

    all_samples = []
    samples_by_class = defaultdict(list)

    for sf in subfolders:
        folder_path = os.path.join(root_dir, sf)
        mask_paths = glob.glob(os.path.join(folder_path, "*_mask.png"))

        for mp in mask_paths:
            img_path = mp.replace("_mask", "")
            if os.path.exists(img_path):
                if sf == "normal":
                    class_id = 0
                elif sf == "benign":
                    class_id = 1
                elif sf == "malignant":
                    class_id = 2

                sample = (img_path, mp, class_id)
                all_samples.append(sample)
                samples_by_class[class_id].append(sample)

    train_samples = []
    val_samples = []

    if stratify:
        for class_id, samples in samples_by_class.items():
            random.shuffle(samples)
            val_count = int(len(samples) * val_ratio)

            val_samples.extend(samples[:val_count])
            train_samples.extend(samples[val_count:])

            print(f"Class {class_id}: {len(samples)} total, {len(samples) - val_count} train, {val_count} validation")
    else:
        random.shuffle(all_samples)
        val_count = int(len(all_samples) * val_ratio)
        val_samples = all_samples[:val_count]
        train_samples = all_samples[val_count:]

    train_dataset = MultiClassBUSIDataset(
        root_dir,
        image_size=image_size,
        samples=train_samples,
        include_normal=include_normal
    )

    val_dataset = MultiClassBUSIDataset(
        root_dir,
        image_size=image_size,
        samples=val_samples,
        include_normal=include_normal
    )

    print(f"Total: {len(all_samples)} samples")
    print(f"Train: {len(train_samples)} samples")
    print(f"Validation: {len(val_samples)} samples")

    return train_dataset, val_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
      self.bn1 = nn.BatchNorm2d(out_channels)

      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
      self.bn2 = nn.BatchNorm2d(out_channels)

      self.relu = nn.ReLU()

    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)

      x = self.conv2(x)
      x = self.bn2(x)
      x = self.relu(x)
      return x


class Encoder(nn.Module): #Make shape of the output ft maps remain same and input ft maps
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = Block(in_channels, out_channels)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, inputs):
    x = self.conv1(inputs)
    p = self.pool(x)
    return x, p


class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    #expansive path
    # "up-convolution" halves # of feature channels
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
    self.conv = Block(out_channels+out_channels, out_channels)

  def forward(self, inputs, skip):
    x = self.up(inputs)
    x = torch.cat([x, skip], axis = 1)
    x = self.conv(x)
    return x


class Unet(nn.Module):
  def __init__(self, in_channels=1, out_channels=3):
    super().__init__()
    """Encoder"""
    self.e1 = Encoder(in_channels, 64)
    self.e2 = Encoder(64, 128)
    self.e3 = Encoder(128, 256)
    self.e4 = Encoder(256, 512)

    """Bottleneck"""
    self.b = Block(512, 1024)

    """Decoder"""
    self.d1 = Decoder(1024, 512)
    self.d2 = Decoder(512, 256)
    self.d3 = Decoder(256, 128)
    self.d4 = Decoder(128, 64)

    """Classifier"""
    self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

  def forward(self, inputs):
    """Encoder"""
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)

    """Bottleneck"""
    b = self.b(p4)

    """Decoder"""
    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)

    """Classifier"""
    outputs = self.outputs(d4)

    return outputs



"""def multi_class_dice_loss(pred, target, smooth=1):
  """
    Computes Dice Loss for multi-class segmentation.
    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: One-hot encoded ground truth (batch_size, C, H, W).
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
  pred = F.softmax(pred, dim=1) # apply softmax to predictions along the channel dimension
  num_classes = pred.shape[1] # get the number of classes
  dice = 0 # initialize the dice loss accumulator

  # iterate over classes
  for c in range(num_classes):
    pred_c = pred[:, c, :, :] # extract predictions for class c (batch_size, H, W)
    target_c = (target == c).float() # create a binary mask for class c (batch_size, H, W)
    intersection = (pred_c * target_c).sum(dim=(1, 2)) # calculate intersection
    union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) # calculate union
    dice += (2. * intersection + smooth) / (union + smooth) # calculate dice coefficient

  return 1 - dice.mean() / num_classes # average dice loss across classes"""

train_dataset, val_dataset = create_train_val_datasets(
    root_dir='/content/content/kaggle_dataset_3/train',
    val_ratio=0.2,
    image_size=(256, 256),
    include_normal=True,  # Set to False if you don't want normal cases
    stratify=True         # Maintain class distribution in splits
)

# Create data loaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
from torchvision.utils import make_grid
import random
from IPython.display import clear_output, display
import time

def train_multiclass_unet(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4,
                         class_weights=None, device=None, project_name="unet-segmentation",
                         experiment_name="unet-training", visualize_sample_ids=None,
                         use_wandb=True, display_freq=1, interactive=True):
    """
    Training function for multi-class UNet segmentation model with visualization

    Args:
        model: MultiClassUNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        class_weights: Optional tensor of weights for each class to handle class imbalance
        device: Device to train on (cuda or cpu)
        project_name: Name for wandb project
        experiment_name: Name for this specific experiment run
        visualize_sample_ids: List of three sample indices to visualize (if None, random samples are chosen)
        use_wandb: Whether to use Weights & Biases logging (default: True)
        display_freq: How often to update the interactive plots in epochs (default: 1)
        interactive: Whether to display real-time plots in the notebook (default: True)
    """
    # Initialize device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Initialize loss function, optimizer and scheduler
    if class_weights is not None:
        # class_weights = torch.tensor([1,1,5], dtype = torch.float32).to(device)
        class_weights = class_weights.to(device)
        # criterion = nn.CrossEntropyLoss(weight = class_weights)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(project=project_name, name=experiment_name, config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "weight_decay": 1e-5,
            "device": str(device),
            "class_weights": class_weights.cpu().numpy().tolist() if class_weights is not None else None
        })

    # Create a palette for visualization (background, benign, malignant)
    class_colors = torch.tensor([
        [0, 0, 0],       # Background - black
        [0, 180, 255],   # Benign - blue
        [255, 77, 77]    # Malignant - red
    ], dtype=torch.uint8)

    # Select visualization samples if not provided
    if visualize_sample_ids is None:
        # Get a deterministic subset of validation data for visualization
        val_dataset_size = len(val_loader.dataset)
        visualize_sample_ids = random.sample(range(val_dataset_size), 3)

    # Extract these samples for consistent visualization
    vis_samples = []
    for idx in visualize_sample_ids:
        # Get sample directly from dataset to avoid batch issues
        sample_data, sample_target = val_loader.dataset[idx]
        vis_samples.append((sample_data.unsqueeze(0), sample_target.unsqueeze(0)))

    # Set up interactive plotting if enabled
    if interactive:
        plt.ion()  # Turn on interactive mode
        # Create figure with 4 rows (metrics + 3 samples)
        live_fig, live_axs = plt.subplots(4, 3, figsize=(18, 16))
        live_fig.tight_layout(pad=3.0)

        # Configure metrics plots in top row
        live_axs[0, 0].set_title('Training & Validation Loss')
        live_axs[0, 0].set_xlabel('Epochs')
        live_axs[0, 0].set_ylabel('Loss')
        live_axs[0, 0].grid(True)

        live_axs[0, 1].set_title('Dice Coefficients')
        live_axs[0, 1].set_xlabel('Epochs')
        live_axs[0, 1].set_ylabel('Dice Score')
        live_axs[0, 1].grid(True)

        live_axs[0, 2].set_title('Learning Rate')
        live_axs[0, 2].set_xlabel('Epochs')
        live_axs[0, 2].set_ylabel('LR')
        live_axs[0, 2].grid(True)

    train_losses = []
    val_losses = []
    dice_scores = {
        "dice_bg": [],
        "dice_benign": [],
        "dice_malignant": []
    }
    lr_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Only log to wandb, avoid printing to console which clutters the notebook
            if batch_idx % 10 == 0 and use_wandb:
                wandb.log({
                    "batch": batch_idx + epoch * len(train_loader),
                    "batch_loss": loss.item(),
                })

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {"dice_bg": 0, "dice_benign": 0, "dice_malignant": 0}

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)

                for class_idx, class_name in enumerate(['bg', 'benign', 'malignant']):
                    pred_class = (preds == class_idx).float()
                    target_class = (target == class_idx).float()
                    intersection = (pred_class * target_class).sum().item()
                    union = pred_class.sum().item() + target_class.sum().item()

                    if union > 0:
                        dice = (2 * intersection) / (union + 1e-7)
                        val_metrics[f"dice_{class_name}"] += dice

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        for metric in val_metrics:
            val_metrics[metric] /= len(val_loader)
            dice_scores[metric].append(val_metrics[metric])

        # Generate visualizations for wandb
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for i, (sample_data, sample_target) in enumerate(vis_samples):
            sample_data = sample_data.to(device)
            sample_target = sample_target.to(device)

            # Get model prediction
            with torch.no_grad():
                sample_output = model(sample_data)
                sample_pred = torch.argmax(sample_output, dim=1)

            # Convert to colored segmentation maps
            def class_to_color(tensor):
                colored = torch.zeros((tensor.shape[0], 3, tensor.shape[1], tensor.shape[2]),
                                      dtype=torch.uint8)
                for cls_idx in range(3):  # 3 classes
                    mask = (tensor == cls_idx)
                    for channel in range(3):  # RGB
                        colored[:, channel][mask] = class_colors[cls_idx][channel]
                return colored

            # Create colored versions of ground truth and prediction
            colored_gt = class_to_color(sample_target)
            colored_pred = class_to_color(sample_pred)

            # Convert tensors to numpy for visualization
            img = sample_data[0].cpu().permute(1, 2, 0).numpy()

            # Normalize image for display
            if img.shape[2] == 1:  # Grayscale image
                img = np.repeat(img, 3, axis=2)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            gt = colored_gt[0].cpu().permute(1, 2, 0).numpy() / 255.0
            pred = colored_pred[0].cpu().permute(1, 2, 0).numpy() / 255.0

            # Plot in the figure
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Sample {visualize_sample_ids[i]}: Input")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt)
            axes[i, 1].set_title(f"Sample {visualize_sample_ids[i]}: Ground Truth")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred)
            axes[i, 2].set_title(f"Sample {visualize_sample_ids[i]}: Prediction")
            axes[i, 2].axis('off')

        plt.tight_layout()

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice_bg": val_metrics["dice_bg"],
                "dice_benign": val_metrics["dice_benign"],
                "dice_malignant": val_metrics["dice_malignant"],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "sample_predictions": wandb.Image(fig)
            })

        # Store learning rate history for plotting
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Update LR scheduler
        scheduler.step(val_loss)

        plt.close(fig)  # Close the figure to free memory

        # Update interactive plots if enabled
        if interactive and (epoch % display_freq == 0 or epoch == num_epochs - 1):
            # Clear previous output for cleaner display
            clear_output(wait=True)

            # Update loss plot
            live_axs[0, 0].clear()
            live_axs[0, 0].plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss')
            live_axs[0, 0].plot(range(len(val_losses)), val_losses, 'r-', label='Val Loss')
            live_axs[0, 0].legend()
            live_axs[0, 0].set_title('Training & Validation Loss')
            live_axs[0, 0].set_xlabel('Epochs')
            live_axs[0, 0].set_ylabel('Loss')
            live_axs[0, 0].grid(True)

            # Update Dice coefficients plot
            live_axs[0, 1].clear()
            for dice_type, color, marker in zip(
                dice_scores.keys(),
                ['g-', 'b-', 'r-'],
                ['o', 's', '^']
            ):
                class_name = dice_type.split('_')[1]
                live_axs[0, 1].plot(
                    range(len(dice_scores[dice_type])),
                    dice_scores[dice_type],
                    color+marker,
                    label=f'Dice {class_name.capitalize()}'
                )
            live_axs[0, 1].legend()
            live_axs[0, 1].set_title('Dice Coefficients')
            live_axs[0, 1].set_xlabel('Epochs')
            live_axs[0, 1].set_ylabel('Dice Score')
            live_axs[0, 1].grid(True)

            # Update learning rate plot
            live_axs[0, 2].clear()
            live_axs[0, 2].plot(range(len(lr_history)), lr_history, 'g-o')
            live_axs[0, 2].set_title('Learning Rate')
            live_axs[0, 2].set_xlabel('Epochs')
            live_axs[0, 2].set_ylabel('Learning Rate')
            live_axs[0, 2].set_yscale('log')  # Log scale for better visualization
            live_axs[0, 2].grid(True)

            # Display all three samples with their predictions
            for i, (sample_data, sample_target) in enumerate(vis_samples):
                # Get prediction for the sample
                model.eval()
                with torch.no_grad():
                    sample_data = sample_data.to(device)
                    sample_output = model(sample_data)
                    sample_pred = torch.argmax(sample_output, dim=1)

                # Convert tensors to displayable images
                img = sample_data[0].cpu().permute(1, 2, 0).numpy()
                # Normalize image for display
                if img.shape[2] == 1:  # Grayscale image
                    img = np.repeat(img, 3, axis=2)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # Create colored versions of ground truth and prediction
                def class_to_color(tensor):
                    colored = torch.zeros((tensor.shape[0], 3, tensor.shape[1], tensor.shape[2]),
                                        dtype=torch.uint8)
                    for cls_idx in range(3):  # 3 classes
                        mask = (tensor == cls_idx)
                        for channel in range(3):  # RGB
                            colored[:, channel][mask] = class_colors[cls_idx][channel]
                    return colored

                # Create colored versions
                colored_gt = class_to_color(sample_target)[0].cpu().permute(1, 2, 0).numpy() / 255.0
                colored_pred = class_to_color(sample_pred)[0].cpu().permute(1, 2, 0).numpy() / 255.0

                # Display input image - use row i+1 (rows 1,2,3 for the three samples)
                row_idx = i + 1  # First row is for metrics, so samples start at row 1

                live_axs[row_idx, 0].clear()
                live_axs[row_idx, 0].imshow(img)
                live_axs[row_idx, 0].set_title(f'Sample {visualize_sample_ids[i]}: Input')
                live_axs[row_idx, 0].axis('off')

                # Display ground truth
                live_axs[row_idx, 1].clear()
                live_axs[row_idx, 1].imshow(colored_gt)
                live_axs[row_idx, 1].set_title(f'Ground Truth')
                live_axs[row_idx, 1].axis('off')

                # Display prediction
                live_axs[row_idx, 2].clear()
                live_axs[row_idx, 2].imshow(colored_pred)
                live_axs[row_idx, 2].set_title(f'Prediction')
                live_axs[row_idx, 2].axis('off')

            # Show the updated figure
            live_fig.suptitle(f'Training Progress - Epoch {epoch}/{num_epochs}', fontsize=16)
            live_fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])  # Make room for suptitle
            display(live_fig)
            plt.pause(0.1)  # Small pause to update the display

    # Plot final training curves
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot loss curves
    axs[0].plot(range(num_epochs), train_losses, label='Train Loss')
    axs[0].plot(range(num_epochs), val_losses, label='Val Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Training and Validation Loss')
    axs[0].grid(True)

    # Plot Dice scores
    for dice_type in dice_scores:
        class_name = dice_type.split('_')[1]
        axs[1].plot(range(num_epochs), dice_scores[dice_type],
                   label=f'Dice {class_name.capitalize()}')

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Dice Score')
    axs[1].legend()
    axs[1].set_title('Validation Dice Scores by Class')
    axs[1].grid(True)

    plt.tight_layout()
    if use_wandb:
        wandb.log({"final_metrics": wandb.Image(fig)})
    plt.close(fig)

    # Finish wandb run if enabled
    if use_wandb:
        wandb.finish()

    # Turn off interactive mode
    if interactive:
        plt.ioff()
        plt.close(live_fig)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'dice_scores': dice_scores
    }

model = Unet(in_channels=1, out_channels=3).to(device)

# Calculate class weights to handle class imbalance
class_counts = torch.zeros(3)
for _, mask in train_dataset:
    # Get unique class values in the mask
    unique_classes = torch.unique(mask)
    for class_idx in unique_classes.tolist():
        if 0 <= class_idx < 3:  # Only consider classes 0, 1, 2
            class_counts[class_idx] += (mask == class_idx).sum().item()

# Inverse frequency weighting
class_weights = 1.0 / (class_counts + 1e-8)
# Normalize weights
class_weights = class_weights / class_weights.sum() * 3

print(f"Class weights: {class_weights}")

training_history = train_multiclass_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=1e-4,  # Slightly lower learning rate for stability
    class_weights=class_weights, # Use class weights to handle imbalance
)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


def visualize_multiclass_prediction(model, data_loader, num_samples=4, save_dir=None):
    """
    Visualizes model predictions alongside input images and ground truth masks for multi-class segmentation.

    Args:
        model: The trained UNet model
        data_loader: DataLoader containing validation or test data
        num_samples: Number of samples to visualize
        save_dir: Directory to save the visualizations (if None, just displays them)
    """
    model.eval()

    colors = np.array([[0, 0, 0],       # Background - Black
                      [0, 0.7, 1],      # Benign - Blue
                      [1, 0.3, 0.3]])   # Malignant - Red

    segmentation_cmap = ListedColormap(colors)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    samples_processed = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            batch_size = images.shape[0]
            for i in range(batch_size):
                if samples_processed >= num_samples:
                    break

                img = images[i].detach().cpu().numpy().squeeze()
                mask = masks[i].detach().cpu().numpy()
                pred = preds[i].detach().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Input Ultrasound')
                axes[0].axis('off')

                axes[1].imshow(mask, cmap=segmentation_cmap, vmin=0, vmax=2)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred, cmap=segmentation_cmap, vmin=0, vmax=2)
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(3)]
                labels = ['Background', 'Benign', 'Malignant']
                fig.legend(patches, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

                # Calculate class-wise Dice scores
                dice_scores = []
                for class_idx in range(3):
                    pred_class = (pred == class_idx).astype(np.float32)
                    mask_class = (mask == class_idx).astype(np.float32)

                    intersection = np.sum(pred_class * mask_class)
                    union = np.sum(pred_class) + np.sum(mask_class)

                    dice = (2. * intersection + 1e-7) / (union + 1e-7)
                    dice_scores.append(dice)

                fig.suptitle(f'Dice Scores - Background: {dice_scores[0]:.3f}, '
                            f'Benign: {dice_scores[1]:.3f}, Malignant: {dice_scores[2]:.3f}')

                plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for legend

                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, f'sample_{samples_processed}.png'),
                                bbox_inches='tight', dpi=150)
                    plt.close()
                else:
                    plt.show()

                samples_processed += 1

            if samples_processed >= num_samples:
                break

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2) # shuffle to visualize different images

visualize_multiclass_prediction(
    model, val_loader, num_samples=20
)

import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

class BUSITestDataset(Dataset):
    """Dataset for BUSI test images."""
    def __init__(self, test_dir, image_size=(256, 256)):
        self.test_dir = test_dir
        self.image_size = image_size

        # Get all image files
        self.image_files = [f for f in os.listdir(test_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        self.image_files.sort()  # Sort for consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.test_dir, image_name)

        # Get image ID (filename without extension)
        image_id = os.path.splitext(image_name)[0]

        # Read image in original size
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_size = original_image.shape  # Store original size

        # Resize for model input
        image = cv2.resize(original_image, self.image_size, interpolation=cv2.INTER_AREA)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension

        return {
            'image': image_tensor,
            'image_id': image_id,
            'original_h': original_size[0],  # Store original size for correct RLE encoding
            'original_w': original_size[1]
        }

def rle_encode_mask(mask):
    """Run-length encode a binary mask."""
    if np.sum(mask) == 0:
        return ''

    pixels = mask.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = np.concatenate([[0], runs])

    run_lengths = []
    for i in range(len(runs) - 1):
        if pixels[runs[i]] == 1:
            start = runs[i] + 1  # 1-indexed
            length = runs[i + 1] - runs[i]
            run_lengths.extend([start, length])

    return ' '.join(str(x) for x in run_lengths)

def combined_encode(masks_dict, delimiter="~"):
    """
    Encode multiple class masks into a single string.

    Args:
        masks_dict: Dictionary mapping class_id to binary mask
        delimiter: String to use as delimiter between class encodings

    Returns:
        Combined encoded string in format "class_id:rle~class_id:rle~..."
    """
    if not masks_dict:
        return ""

    encoded_parts = []

    for class_id, mask in masks_dict.items():
        rle = rle_encode_mask(mask)
        if rle:  # Only include non-empty masks
            encoded_parts.append(f"{class_id}:{rle}")

    return delimiter.join(encoded_parts)

def generate_submission(model, test_dir, output_file, batch_size=4, device=None, image_size=(256, 256)):
    """
    Generate a submission file from model predictions on test images.

    Args:
        model: Trained PyTorch model
        test_dir: Directory containing test images
        output_file: Path to save the submission CSV
        batch_size: Batch size for inference
        device: Device to run inference on
        image_size: Size to resize images to
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test dataset and loader
    test_dataset = BUSITestDataset(test_dir, image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Results dictionary
    results = {
        'ID': [],
        'encoded_pixels': []
    }

    # Process each batch
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            original_hs = batch['original_h']  # [B, 2] tensor with original H, W
            original_ws = batch['original_w']  # [B, 2] tensor with original H, W

            # Forward pass
            outputs = model(images)

            # Process predictions
            batch_size = images.shape[0]

            # For multi-class segmentation (assuming outputs have shape [B, C, H, W])
            is_multiclass = outputs.shape[1] > 1

            if is_multiclass:
                # Get class predictions for each pixel
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # [B, H, W]

                for i in range(batch_size):
                    image_id = image_ids[i]
                    pred = predictions[i]  # [H, W]
                    orig_h, orig_w = original_hs[i].item(), original_ws[i].item()

                    # Resize prediction back to original image size before RLE encoding
                    # CRITICAL: Use nearest neighbor interpolation for masks
                    pred_resized = cv2.resize(
                        pred.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int32)

                    # Create mask dictionary for this image
                    masks_dict = {}
                    for class_id in range(1, outputs.shape[1]):
                        # Create binary mask for this class using the RESIZED prediction
                        binary_mask = (pred_resized == class_id).astype(np.uint8)

                        # Only add non-empty masks
                        if np.sum(binary_mask) > 0:
                            masks_dict[class_id] = binary_mask

                    # Encode masks (will return empty string if no masks)
                    encoded_pixels = combined_encode(masks_dict)

                    # Add to results (even if encoded_pixels is empty)
                    results['ID'].append(image_id)
                    results['encoded_pixels'].append(encoded_pixels)

            else:
                # Binary segmentation - apply sigmoid and threshold
                predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5

                for i in range(batch_size):
                    image_id = image_ids[i]
                    pred = predictions[i, 0]  # [H, W]
                    orig_h, orig_w = original_sizes[i]

                    # Resize prediction back to original image size
                    pred_resized = cv2.resize(
                        pred.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)

                    # Create mask dictionary for classes 1 and 2
                    masks_dict = {}
                    for class_id in [1, 2]:  # Both benign and malignant
                        binary_mask = pred_resized.astype(np.uint8)

                        # Only add non-empty masks
                        if np.sum(binary_mask) > 0:
                            masks_dict[class_id] = binary_mask

                    # Encode masks (will return empty string if no masks)
                    encoded_pixels = combined_encode(masks_dict)

                    # Add to results (even if encoded_pixels is empty)
                    results['ID'].append(image_id)
                    results['encoded_pixels'].append(encoded_pixels)

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Make sure there are no NULL/None values - replace with empty strings
    submission_df['encoded_pixels'] = submission_df['encoded_pixels'].fillna('')
    submission_df.loc[submission_df['encoded_pixels'] == '', 'encoded_pixels'] = '<empty>'

    # Sort by image_id for consistency
    submission_df = submission_df.sort_values('ID')

    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(f"Total entries: {len(submission_df)}")
    print(f"Empty predictions: {(submission_df['encoded_pixels'] == '<empty>').sum()}")

    # Validate submission
    if submission_df.isnull().any().any():
        print("WARNING: Submission contains NULL values!")
    else:
        print("Validation passed: No NULL values in submission.")

    return submission_df

generate_submission(
    model=model,
    test_dir='/content/content/kaggle_dataset_3/test',
    output_file='/content/kaggle_A2.csv',
    batch_size=16,
    device=device
)

from google.colab import files
files.download("/content/kaggle_A2.csv")
