import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from dataset import get_dataloaders
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    model     = build_model(num_classes=10, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(args.model_dir, exist_ok=True)
    best_val_acc = 0.0

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "freeze_backbone": True,
            "optimizer":   "Adam",
            "scheduler":   "StepLR(step=5, gamma=0.5)",
        })

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(args.model_dir, "mobilenet_finetuned.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> checkpoint saved (val_acc={best_val_acc:.3f})")

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(ckpt_path)
        # view results: mlflow ui --port 5001
        print(f"\nTraining complete. Best val_acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="../data")
    parser.add_argument("--model_dir",  default="../models")
    parser.add_argument("--experiment", default="cifar10-mobilenetv2")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)