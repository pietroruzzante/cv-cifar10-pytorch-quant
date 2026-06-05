import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import mlflow

from dataset import get_dataloaders, CIFAR10_CLASSES
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(args):
    device = get_device()
    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    model = build_model(num_classes=10, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds  = model(images).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES))

    _plot_confusion_matrix(all_labels, all_preds, args.output)

    if args.run_id:
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metric("test_acc", accuracy)
            mlflow.log_artifact(args.output)
            print(f"Logged to MLflow run {args.run_id}")


def _plot_confusion_matrix(labels, preds, output_path: str):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")

    thresh = 0.5
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="../models/mobilenet_finetuned.pth")
    parser.add_argument("--data_dir",    default="../data")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--output",      default="../models/confusion_matrix.png")
    parser.add_argument("--run_id",      default=None, help="MLflow run ID to log results into")
    args = parser.parse_args()
    evaluate(args)
