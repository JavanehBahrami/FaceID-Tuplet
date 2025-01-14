import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import pdb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from box import Box
from loguru import logger



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n used Device: {device}")
print("-----------------------------------------")

# Step 1: Custom Dataset and DataLoader for Tuplet Loss with Multiple Negative IDs
class TupletDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_negatives=10):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_images = {cls: os.listdir(os.path.join(data_dir, cls)) for cls in self.classes}
        self.num_negatives = num_negatives
        print(f"self.num_negatives: {self.num_negatives}")
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        # Select the anchor class and image
        anchor_class = self.classes[index]
        anchor_image_path = random.choice(self.class_to_images[anchor_class])
        anchor_image = Image.open(os.path.join(self.data_dir, anchor_class, anchor_image_path)).convert('RGB')

        # Select a positive sample from the same class
        positive_image_path = random.choice(self.class_to_images[anchor_class])
        positive_image = Image.open(os.path.join(self.data_dir, anchor_class, positive_image_path)).convert('RGB')

        # Select negative samples from different classes
        negative_classes = random.sample([cls for cls in self.classes if cls != anchor_class], self.num_negatives)
        negatives = []
        for neg_class in negative_classes:
            neg_image_path = random.choice(self.class_to_images[neg_class])
            negative_image = Image.open(os.path.join(self.data_dir, neg_class, neg_image_path)).convert('RGB')
            negatives.append(self.transform(negative_image) if self.transform else negative_image)
        
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
        
        negatives = torch.stack(negatives) if negatives else torch.empty(0)
        return anchor_image, positive_image, negatives


# Step 4: Metrics Calculation Function
def _calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = sum(y1 == y2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(cm) == 1:
            if y_true[0] == 1:
                tp = cm[0, 0]
            else:
                tn = cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return precision, recall, f1, tpr, fpr, fnr, accuracy


def load_model(config):
    # Step 5: Model Setup
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(config.device)
    logger.info("pre-trained model is loaded successfully.")

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    return model


# Step 3: Tuplet Loss Function (Updated for Online Hard Negative Mining)
def tuplet_loss(anchor_emb, positive_emb, negative_embs, margin=0.2, k=2):
    # Calculate positive distances
    positive_dist = F.pairwise_distance(anchor_emb, positive_emb)
    
    # Calculate negative distances
    if negative_embs.numel() > 0:  # Ensure negatives are not empty
        neg_dist = torch.cdist(anchor_emb, negative_embs.view(-1, anchor_emb.size(1))).view(anchor_emb.size(0), -1)
        # Select the k hardest negatives
        top_k_negatives, _ = torch.topk(neg_dist, k, dim=1, largest=False)
        hardest_negative_dist = torch.mean(top_k_negatives, dim=1)
    else:
        # Default to a high distance when no negatives exist
        hardest_negative_dist = torch.tensor(float('inf')).to(anchor_emb.device)
    
    # Calculate the loss
    loss = torch.mean(F.relu(positive_dist - hardest_negative_dist + margin))
    
    # Debugging with detached tensors
    # print(f"Positive Distances: {positive_dist[:5].detach().cpu().numpy()}")
    # print(f"Hardest Negative Distances: {hardest_negative_dist[:5].detach().cpu().numpy()}")
    
    return loss


def _save_checkpoint(checkpoint_abs_path, saved_params):
    epoch = saved_params['epoch']
    total_loss = saved_params['total_loss']
    val_acc = saved_params['val_acc']
    val_tpr = saved_params['val_tpr']
    val_fpr = saved_params['val_fpr']
    val_fnr = saved_params['val_fnr']
    val_prec = saved_params['val_prec']
    val_rec = saved_params['val_rec']
    val_f1 = saved_params['val_f1']

    # Save checkpoint
    checkpoint_path = os.path.join(
        checkpoint_abs_path,
        f"checkpoint_epoch{epoch}_loss{total_loss / len(dataloader):.4f}_"
        f"valAcc{val_acc:.4f}_valTPR{val_tpr:.4f}_valFPR{val_fpr:.4f}_valFNR{val_fnr:.4f}.pth"
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': total_loss / len(dataloader),
        'val_acc': val_acc,
        'val_prec': val_prec,
        'val_rec': val_rec,
        'val_f1': val_f1,
        'val_tpr': val_tpr,
        'val_fpr': val_fpr,
        'val_fnr': val_fnr
    }, checkpoint_path)
    return


def train_model(config, model, checkpoint_abs_path):
    print("-------    start training tuplet loss    ----------")
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0
        y_true_train, y_pred_train = [], []

        for anchor, positive, negatives in dataloader:
            anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
            optimizer.zero_grad()

            with autocast():
                # Forward pass for anchor and positive embeddings
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                
                # Forward pass for negatives
                if negatives.size(0) > 0:
                    negative_embs = model(negatives.view(-1, 3, config.transform.resize[0], config.transform.resize[1])).view(negatives.size(0), negatives.size(1), -1)
                    # pdb.set_trace()
                    
                    # Call the tuplet_loss function to calculate the loss
                    loss = tuplet_loss(anchor_emb,
                                       positive_emb,
                                       negative_embs,
                                       margin=config.loss_margin,
                                       k=config.loss_k)
                    
                    # Calculate metrics
                    pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                    neg_dist = F.pairwise_distance(anchor_emb.unsqueeze(1).expand(-1, negatives.size(1), -1), negative_embs)
                    y_true_train.extend([1] * len(pos_dist))
                    y_true_train.extend([0] * neg_dist.numel())
                    predictions = (pos_dist < torch.min(neg_dist, dim=1).values).cpu().numpy().astype(int)
                    y_pred_train.extend(predictions)
                    y_pred_train.extend([0] * neg_dist.numel())
                else:
                    continue  # Skip batch if no negatives

            # Backpropagation and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Evaluate metrics after epoch
        with torch.no_grad():
            val_prec, val_rec, val_f1, val_tpr, val_fpr, val_fnr, val_acc = _calculate_metrics(y_true_train,
                                                                                               y_pred_train)

        # Logging metrics
        print(f"Epoch {epoch}: Loss={total_loss / len(dataloader):.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, "
            f"F1={val_f1:.4f}, TPR={val_tpr:.4f}, FPR={val_fpr:.4f}, FNR={val_fnr:.4f}")

        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch}, Loss={total_loss / len(dataloader):.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, "
                    f"Rec={val_rec:.4f}, F1={val_f1:.4f}, TPR={val_tpr:.4f}, FPR={val_fpr:.4f}, FNR={val_fnr:.4f}\n")

        # Learning rate scheduler step
        if scheduler:
            scheduler.step()

        saved_params = {
            'epoch': epoch,
            'total_loss': total_loss,
            'val_acc': val_acc,
            'val_tpr': val_tpr,
            'val_fpr': val_fpr,
            'val_fnr': val_fnr,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_f1': val_f1
        }
        _save_checkpoint(checkpoint_abs_path, saved_params)

    return


if __name__ == "__main__":
    with open("config.yaml", "r") as config_file:
        config_dict = yaml.safe_load(config_file)
        config = Box(config_dict)


    # data_dir = config.data_dir
    # batch_size = config.batch_size
    # num_negatives = config.num_negatives

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Step 2: Initialize dataset and dataloader with multiple negatives
    print(f"batch_size: {config.batch_size}")
    print(f"num_negatives: {config.num_negatives}")

    dataset = TupletDataset(data_dir=config.data_dir,
                            transform=transform,
                            num_negatives=config.num_negatives)

    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=8, 
                            pin_memory=True)
    print("------   dataloader is created   -------------")

    model = load_model(config)


    # Step 5: Optimizer with weight decay and learning rate scheduler
    # pdb.set_trace()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.lr_step_size,
                                          gamma=config.lr_gamma)
    scaler = GradScaler()

    # Step 6: Training Loop
    print(f"number of epochs: {config.num_epochs}")
    print(f"checkpoint number: {config.train_index}")
    checkpoint_abs_path = config.checkpoint_abs_path

    # Check if the path exists, if not, create it
    if not os.path.exists(checkpoint_abs_path):
        os.makedirs(checkpoint_abs_path)
        print(f"Created directory: {checkpoint_abs_path}")
    else:
        print(f"Directory already exists: {checkpoint_abs_path}")

    checkpoint_abs_path = f"{checkpoint_abs_path}/test{config.train_index}_tuplet_loss"
    log_file = os.path.join(checkpoint_abs_path, f"training_log{config.train_index}.txt")
    os.makedirs(checkpoint_abs_path, exist_ok=True)

    with open(log_file, "a") as log:
        log.write("Epoch,Train_Loss,Val_Acc,Val_Prec,Val_Rec,Val_F1,Val_TPR,Val_FPR,Val_FNR\n")
    
    train_model(config, model, checkpoint_abs_path)
    logger.info("<><><>  Training is Done   <><><>")



