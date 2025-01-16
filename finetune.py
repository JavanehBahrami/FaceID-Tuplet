import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torch.cuda.amp import GradScaler, autocast
from box import Box
from loguru import logger
from utils.custom_data_generator import TupletDataset
from utils.evaluation_metrics import calculate_metrics
from utils.loss_utils import tuplet_loss


def load_model(device):
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    logger.info("pre-trained model is loaded successfully.")

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    return model


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
    logger.info("-------    start training tuplet loss    ----------")
    for epoch in range(1, config.train.num_epochs + 1):
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
                    negative_embs = model(negatives.view(-1, 3, config.train.transform.resize[0], config.train.transform.resize[1])).view(negatives.size(0), negatives.size(1), -1)
                    
                    # Call the tuplet_loss function to calculate the loss
                    loss = tuplet_loss(anchor_emb,
                                       positive_emb,
                                       negative_embs,
                                       margin=config.train.loss_margin,
                                       k=config.train.loss_k)
                    
                    # Calculate metrics
                    pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                    neg_dist = F.pairwise_distance(anchor_emb.unsqueeze(1).expand(-1,
                                                                                  negatives.size(1), -1),
                                                                                  negative_embs)
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

        with torch.no_grad():
            val_prec, val_rec, val_f1, val_tpr, val_fpr, val_fnr, val_acc = calculate_metrics(y_true_train,
                                                                                              y_pred_train)

        print(
            f"Epoch {epoch}: Loss={total_loss / len(dataloader):.4f}, "
            f"Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, "
            f"F1={val_f1:.4f}, TPR={val_tpr:.4f}, FPR={val_fpr:.4f}, "
            f"FNR={val_fnr:.4f}"
        )

        with open(log_file, "a") as log:
            log.write(
                f"Epoch {epoch}, Loss={total_loss / len(dataloader):.4f}, "
                f"Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, "
                f"F1={val_f1:.4f}, TPR={val_tpr:.4f}, FPR={val_fpr:.4f}, "
                f"FNR={val_fnr:.4f}\n"
            )

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

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Available Device: {device}")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((config.train.transform.resize[0],
                           config.train.transform.resize[1])),
        transforms.ToTensor(),
        transforms.Normalize(config.train.transform.normalize.mean,
                             config.train.transform.normalize.std)
    ])

    dataset = TupletDataset(data_dir=config.train.data_dir,
                            transform=transform,
                            num_negatives=config.train.num_negatives)

    dataloader = DataLoader(dataset,
                            batch_size=config.train.batch_size,
                            shuffle=True,
                            num_workers=8, 
                            pin_memory=True)

    model = load_model(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.learning_rate,
                           weight_decay=config.train.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.train.lr_step_size,
                                          gamma=config.train.lr_gamma)
    scaler = GradScaler()

    checkpoint_abs_path = config.train.checkpoint_abs_path

    checkpoint_abs_path = f"{checkpoint_abs_path}/test{config.train.train_index}_tuplet_loss"
    log_file = os.path.join(checkpoint_abs_path, f"training_log{config.train.train_index}.txt")
    os.makedirs(checkpoint_abs_path, exist_ok=True)

    with open(log_file, "a") as log:
        log.write("Epoch,Train_Loss,Val_Acc,Val_Prec,Val_Rec,Val_F1,Val_TPR,Val_FPR,Val_FNR\n")
    
    train_model(config, model, checkpoint_abs_path)
    logger.info("<><><>  Training is Done   <><><>")
