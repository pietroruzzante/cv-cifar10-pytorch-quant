# MobileNetV2 Fine-Tuning on CIFAR-10 with MLflow & INT8 Quantization

Transfer learning pipeline on CIFAR-10 using MobileNetV2, from training to edge-ready deployment. A two-phase strategy (feature extraction → full fine-tuning) brings test accuracy to **94.7%**. The model is then compressed via INT8 quantization, shrinking from 8.51 MB to **2.31 MB (-73%)** in ONNX format — ready for deployment via TensorRT or OpenVINO. Unstructured pruning experiments show that MobileNetV2's architecture tolerates up to 30% sparsity (-2.2% accuracy) before collapsing, confirming quantization as the right compression path. All experiments tracked with MLflow.

**Stack:** Python · PyTorch · MLflow · ONNX Runtime · scikit-learn

---

## Why these choices

**MobileNetV2 over ResNet:** uses depthwise separable convolutions, which factorize a standard convolution into a spatial step and a channel-mixing step. This reduces parameters from ~25M (ResNet50) to ~3.4M with comparable accuracy — directly relevant to embedded targets where flash memory and compute budget are constrained.

**Two-phase training strategy:**

- **Phase 1 — Feature extraction (5 epochs, lr=1e-3):** the backbone is frozen. Only the classifier head (~12K params out of 3.4M) is trained. The pretrained ImageNet weights encode generic features (edges, textures, shapes) that transfer well to CIFAR-10. Starting here avoids destroying those filters with a high learning rate before the head has converged.

- **Phase 2 — Fine-tuning (10 epochs, lr=1e-4):** the full backbone is unfrozen and trained end-to-end with a learning rate 10x lower, using cosine annealing. This lets the backbone adapt its filters to CIFAR-10 without catastrophic forgetting of the pretrained representations. The accuracy jump from ~78% to ~95% happens entirely in this phase.

**INT8 quantization via ONNX Runtime:** the trained model is exported to ONNX (FP32) and quantized dynamically to INT8. Each weight goes from 4 bytes to 1 byte, achieving ~73% model size reduction. ONNX is the standard interchange format for edge runtimes — the same INT8 model can be deployed via TensorRT (NVIDIA Jetson), OpenVINO (Intel), or ONNX Runtime Mobile with no further conversion.

**Unstructured L1 pruning:** individual weights below a threshold are zeroed out globally across all Conv2d and Linear layers. MobileNetV2 tolerates up to 30% sparsity with minimal degradation but collapses beyond 50% — its architecture is already near-optimal with little redundancy to remove, making quantization the preferred compression path.

---

## Project structure

```
├── src/
│   ├── dataset.py      # CIFAR-10 DataLoader, ImageNet normalization, train/val/test splits
│   ├── model.py        # MobileNetV2 with custom 10-class head
│   ├── train.py        # Two-phase training loop with MLflow logging
│   ├── evaluate.py     # Test accuracy, per-class report, confusion matrix
│   ├── quantize.py     # ONNX export + INT8 dynamic quantization + size benchmark
│   ├── prune.py        # Unstructured L1 pruning across sparsity levels
│   └── export_onnx.py  # Standalone FP32 ONNX export with verification
├── Makefile
└── requirements.txt
```

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# train (downloads CIFAR-10 automatically on first run)
make train

# open MLflow UI at http://localhost:5001 — grab the RUN_ID from there
make mlflow
```

```bash
# run evaluation, quantization and pruning (replace with your run ID)
make evaluate RUN_ID=<run_id>
make quantize RUN_ID=<run_id>
make prune    RUN_ID=<run_id>
```

---

## Results

### Accuracy — two-phase training
| Phase | Test Accuracy |
|-------|--------------|
| Phase 1 only (frozen backbone) | 78.1% |
| Phase 1 + 2 (full fine-tuning) | **94.7%** |

### Quantization — model size
| Format | Size |
|--------|------|
| FP32 ONNX | 8.51 MB |
| INT8 ONNX | 2.31 MB (**-73%**) |

### Pruning — accuracy vs sparsity
| Sparsity | Accuracy | Drop |
|----------|----------|------|
| 0% (baseline) | 94.7% | — |
| 30% | 92.5% | -2.2% |
| 50% | 29.5% | -65.2% |
| 70% | 10.0% | -84.7% |

> The sharp accuracy cliff between 30% and 50% confirms that MobileNetV2 has minimal weight redundancy. Quantization is the correct compression path for this architecture.
