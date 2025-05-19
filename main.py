import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import os
from torch.utils.data import random_split
from tqdm import tqdm
from collections import defaultdict
from dataset import LossData
from simple_model import LossModel as LossModel
from losses import Siou_loss_full,loss_MAE,hybrid_loss,hybrid_loss_sum,iou_loss,angle_and_distance_cost,angle_cost

loss_data = LossData("archive/coco128/labels/train2017","archive/coco128/purturbed_labels/train2017")
print(len(loss_data))

train_size = int(0.80 * len(loss_data))
test_size = len(loss_data) - train_size
training_set,validation_set = random_split(loss_data, [train_size, test_size])

def build_loaders(dataset,mode):
    dataloader = DataLoader(dataset=dataset,drop_last=True,batch_size=10,shuffle= (True if mode == "Train" else False))
    return dataloader


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

# Global dict: {file_name: {epoch: pred_tensor}}
prediction_log = defaultdict(dict)

def train_loop(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0.0
    tqdm_object = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch}")

    for idx, (pt, gt, file_name) in enumerate(tqdm_object):
        gt, pt = gt.to(device), pt.to(device)
        output = model(pt)

        loss, logs = Siou_loss_full(output, gt, return_components=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i, name in enumerate(file_name):
            prediction_log[name][epoch] = output[i].detach().cpu()

        total_loss += loss.item()
        tqdm_object.set_postfix(loss=loss.item(), lr=get_lr(optimizer))

        # if idx == 0:
        #     # print("🔍 SIoU Loss Components (Train):")
        #     for k, v in logs.items():
        #         print(f"   {k:>15}: {v:.4f}")

    return total_loss / len(train_loader)


def valid_loop(valid_loader, model, optimizer, epoch):
    model.eval()
    valid_loss = 0.0
    tqdm_object = tqdm(valid_loader, total=len(valid_loader), desc="Validation", dynamic_ncols=False)

    with torch.no_grad():
        for idx, (pt, gt, file_name) in enumerate(tqdm_object):
            gt, pt = gt.to(device), pt.to(device)
            output = model(pt)

            loss, logs = Siou_loss_full(output, gt, return_components=True)

            valid_loss += loss.item()
            tqdm_object.set_postfix(loss=loss.item(), lr=get_lr(optimizer))

            # if idx == 0:
            #     print("🔍 SIoU Loss Components (Validation):")
            #     for k, v in logs.items():
            #         print(f"   {k:>15}: {v:.4f}")

    return valid_loss / len(valid_loader)

def main():
    train_loader = build_loaders(training_set, mode="Train")
    valid_loader = build_loaders(validation_set, mode="Valid")

    model = LossModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.02, lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    epochs = 150
    best_val_loss = float('inf')

    for i in range(epochs):
        print(f"\n📘 Current Epoch: {i}")
        
        train_loss = train_loop(train_loader=train_loader, optimizer=optimizer, model=model, epoch=i)
        torch.save(model.state_dict(), "od_models/temp.pt")

        with torch.no_grad():
            valid_loss = valid_loop(valid_loader=valid_loader, model=model, optimizer=optimizer, epoch=i)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), "od_models/best.pt")

        lr_scheduler.step(valid_loss)

    # ✅ Save prediction logs
    import pickle
    with open("training_predictions2.pkl", "wb") as f:
        pickle.dump(prediction_log, f)

    print("✅ Training complete. Best validation loss:", best_val_loss)

if __name__ == "__main__":
    main()
