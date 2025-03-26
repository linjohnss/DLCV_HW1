import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from utils.dataloader import TrainDataset, TestDataset
from models.resnet import (
    ResNeXt50,
    ResNeXt101,
    ResNeXt101_CBAM,
    ResNeXt101_CBAM_Layer,
)


model_dict = {
    "resnext50": ResNeXt50,
    "resnext101": ResNeXt101,
    "resnext101_cbam": ResNeXt101_CBAM,
    "resnext101_cbam_layer": ResNeXt101_CBAM_Layer,
}


class Trainer:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=args.output_dir)

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(324),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((360, 360)),
            transforms.CenterCrop(324),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

        self.augmix = v2.RandomApply([v2.AugMix()], p=0.5)
        self.cutmix = v2.RandomApply([v2.CutMix(num_classes=100)], p=0.5)
        self.mixup = v2.RandomApply([v2.MixUp(num_classes=100)], p=0.3)

        self.train_loader = DataLoader(
            TrainDataset(args.train_data_dir, self.transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            TrainDataset(args.valid_data_dir, self.transform_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            TestDataset(args.test_data_dir, self.transform_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )

        self.model = model_dict[args.model](
            num_classes=len(self.train_loader.dataset.classes)
        ).to(args.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params/1e6:.10f}M")
        # Label Smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Differential Learning Rates
        params = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if "fc" in n or "cbam" in n
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if "fc" not in n and "cbam" not in n
                ],
                "lr": args.lr / 10,
            },
        ]
        self.optimizer = optim.AdamW(
            params, lr=args.lr, weight_decay=1e-3
        )

        # Cosine Annealing Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-6
        )
        self.scaler = torch.amp.GradScaler()  # Mixed Precision Training
        self.best_accuracy = 0.0

    def train(self):
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            prefetcher = iter(self.train_loader)  # Data Prefetching
            for images, labels in tqdm(
                prefetcher,
                desc=f"Epoch {epoch+1}/{self.args.epochs}"
            ):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                images, labels = self.augmix((images, labels))
                images, labels = self.cutmix((images, labels))
                images, labels = self.mixup((images, labels))

                self.optimizer.zero_grad()
                with torch.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )  # Gradient Clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total += labels.size(0)
                if labels.ndim == 2:
                    labels = labels.argmax(dim=1)
                correct += (outputs.argmax(1) == labels).sum().item()

            self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total
            val_loss, val_acc = self.validate(self.valid_loader)

            self.writer.add_scalar("Loss/Train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            print(
                f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, "
                f"Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}, "
                f"Val Acc {val_acc:.2f}%"
            )

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                ckpt_path = os.path.join(
                    self.args.output_dir,
                    "best_model.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved best model to {ckpt_path}")

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                with torch.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        return total_loss / len(loader), 100.0 * correct / total

    def eval(self):
        ckpt_path = os.path.join(self.args.output_dir, "best_model.pth")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.args.device)
            )
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(
                f"Checkpoint {ckpt_path} not found. Exiting evaluation."
            )
            return

        # === 1. Evaluate on Validation Set (for Accuracy & Confusion Matrix) ===
        self.model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.valid_loader, desc="Validating"):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {acc * 100:.2f}%")

        # Compute confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        # 移除未使用的 num_classes 變數

        # 取得出現次數最多的前 10 類
        import numpy as np
        from collections import Counter

        label_counts = Counter(val_labels)
        top10_labels = [
            label for label, _ in label_counts.most_common(10)
        ]

        # 篩選混淆矩陣為 top10 的 subset
        cm_top10 = cm[np.ix_(top10_labels, top10_labels)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_top10,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[str(i) for i in top10_labels],
            yticklabels=[str(i) for i in top10_labels]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Validation Confusion Matrix (Top 10 Classes)")
        cm_path = os.path.join(
            self.args.output_dir,
            "val_confusion_matrix_top10.png"
        )
        plt.savefig(cm_path)
        plt.close()
        print(
            f"Top-10 validation confusion matrix saved to {cm_path}"
        )

        # === 2. Predict on Test Dataset (Same as before) ===
        predictions = []
        image_names = []

        with torch.no_grad():
            for images, filenames in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.args.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
                image_names.extend(filenames)

        df = pd.DataFrame({
            "image_name": image_names,
            "pred_label": predictions,
        })
        df.to_csv(
            os.path.join(self.args.output_dir, "prediction.csv"),
            index=False
        )
        print("Test prediction saved to prediction.csv.")
