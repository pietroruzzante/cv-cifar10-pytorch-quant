import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import mlflow

from dataset import get_dataloaders
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def count_sparsity(model) -> float:
    zeros = total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            zeros += (module.weight == 0).sum().item()
            total += module.weight.nelement()
    return zeros / total if total > 0 else 0.0


def apply_pruning(model, sparsity: float):
    params_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    # make pruning permanent (remove masks, zero weights are baked in)
    for module, _ in params_to_prune:
        prune.remove(module, "weight")


def run(args):
    device = get_device()
    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    baseline = build_model(num_classes=10, freeze_backbone=False).to(device)
    baseline.load_state_dict(torch.load(args.checkpoint, map_location=device))
    baseline_acc = evaluate_accuracy(baseline, test_loader, device)

    print(f"\n{'='*50}")
    print(f"  Baseline accuracy : {baseline_acc:.4f}")
    print(f"{'='*50}")

    results = {"baseline_acc": baseline_acc}

    for sparsity in args.sparsity_levels:
        model = copy.deepcopy(baseline).to(device)
        apply_pruning(model, sparsity)
        actual_sparsity = count_sparsity(model)
        acc = evaluate_accuracy(model, test_loader, device)
        drop = acc - baseline_acc
        print(f"  Pruning {sparsity*100:.0f}%  | sparsity={actual_sparsity:.3f} | "
              f"acc={acc:.4f} | drop={drop:+.4f}")
        results[f"pruned_{int(sparsity*100)}pct_acc"] = acc

    print(f"{'='*50}\n")

    if args.run_id:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metrics(results)
            print(f"Logged to MLflow run {args.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default="../models/mobilenet_finetuned.pth")
    parser.add_argument("--data_dir",       default="../data")
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--sparsity_levels",type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--run_id",         default=None)
    args = parser.parse_args()
    run(args)
