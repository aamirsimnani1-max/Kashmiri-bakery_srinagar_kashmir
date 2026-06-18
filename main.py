"""
main.py

Training script for an image classifier (Kashmiri Bakery & Sweets)
- Uses PyTorch and torchvision
- Expects a directory structure:
    data/
      train/
        class1/
        class2/
      val/
        class1/
        class2/
- Saves best model and metrics to output_dir

Usage example:
python main.py --data-dir data --epochs 10 --batch-size 32 --lr 1e-3 --num-classes 5 --output-dir outputs

"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classifier for Kashmiri Bakery & Sweets")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes in your dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v2"], help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    return parser.parse_args()


def get_dataloaders(data_dir, image_size, batch_size, workers=4):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    class_to_idx = train_dataset.class_to_idx
    return train_loader, val_loader, class_to_idx


def build_model(model_name, num_classes, pretrained=True):
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=running_corrects / total)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            pbar.set_postfix(loss=running_loss / total, acc=running_corrects / total)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth.tar"):
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(output_dir, "model_best.pth.tar")
        torch.save(state, best_path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Preparing data loaders...")
    train_loader, val_loader, class_to_idx = get_dataloaders(args.data_dir, args.image_size, args.batch_size, args.workers)
    print(f"Found classes: {list(class_to_idx.keys())}")

    print("Building model...")
    model = build_model(args.model, args.num_classes, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, args.output_dir, filename=f"checkpoint_epoch_{epoch}.pth")

        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

        print(f"Epoch {epoch} — train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.2f} minutes. Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
