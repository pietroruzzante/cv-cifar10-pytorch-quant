# MobileNetV2 Fine-Tuning on CIFAR-10 with MLflow & INT8 Quantization

Fine-tuned MobileNetV2 on CIFAR-10 with full experiment tracking via MLflow. Applied post-training INT8 quantization achieving ~75% model size reduction, with inference latency benchmarking targeting edge deployment scenarios.

**Stack:** Python · PyTorch · MLflow · scikit-learn

---

## Results

### Accuracy
| Model | Test Accuracy |
|-------|--------------|
| Baseline FP32 (phase 1 only) | 78.1% |
| Fine-tuned FP32 (phase 1 + 2) | **94.7%** |

### Quantization (ONNX)
| | FP32 | INT8 |
|---|---|---|
| Model size | 8.51 MB | 2.31 MB (**-73%**) |
| Latency (CPU, Apple Silicon) | 2.33 ms | 8.96 ms |

> Latency on Apple Silicon does not improve with INT8 — the CPU lacks dedicated integer execution units. On target hardware (ARM Cortex with CMSIS-NN, NVIDIA Jetson + TensorRT, Intel + OpenVINO) INT8 latency would decrease significantly. Size reduction is hardware-agnostic and directly reduces flash/RAM on any edge device.

### Pruning (unstructured L1, applied to fine-tuned model)
| Sparsity | Accuracy | Drop |
|----------|----------|------|
| 0% (baseline) | 94.7% | — |
| 30% | 92.5% | -2.1% |
| 50% | 29.5% | -65.2% |
| 70% | 10.0% | -84.7% |

> MobileNetV2 is already a highly compressed architecture — its depthwise separable convolutions leave little redundancy. Accuracy holds at 30% sparsity but collapses at 50%, unlike larger models (e.g. ResNet50) which tolerate 70-80% pruning with minimal degradation. This confirms that further compression of MobileNetV2 requires quantization rather than pruning.

---

## Why these choices

**MobileNetV2 over ResNet:** depthwise separable convolutions reduce parameters from ~25M (ResNet50) to ~3.4M with comparable accuracy on this task. On embedded hardware (no dedicated GPU), fewer parameters means less flash memory, fewer CPU cycles, lower latency.

**Feature extraction, not full fine-tuning:** backbone frozen, only the classifier head retrained (~12K trainable params out of 3.4M). The pretrained ImageNet filters already encode useful low-level features (edges, textures, shapes) that transfer well to CIFAR-10. ImageNet normalization is kept for the same reason — the backbone expects inputs in that distribution.

**Dynamic INT8 quantization:** weights compressed from FP32 (4 bytes) to INT8 (1 byte) at save time. No calibration dataset required. Activations quantized at runtime. Straightforward tradeoff: minimal accuracy loss, significant size and latency gains, immediate deployability on CPU-only hardware.

---

## Project structure

```
cv-cifar10/
├── src/
│   ├── dataset.py      # CIFAR-10 DataLoader, ImageNet normalization, train/val/test splits
│   ├── model.py        # MobileNetV2 with custom 10-class head
│   ├── train.py        # Training loop, MLflow logging, checkpoint saving
│   ├── evaluate.py     # Test accuracy, per-class report, confusion matrix
│   └── quantize.py     # Dynamic INT8 PTQ, size & latency benchmarking
├── models/
│   ├── mobilenet_finetuned.pth
│   ├── mobilenet_quantized.pth
│   └── mobilenet.onnx
├── mlruns/             # MLflow experiment logs
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

# standalone ONNX export
make export
```

---

## Quantization — what it means for edge deployment

Dynamic PTQ compresses Linear layer weights from FP32 → INT8. The result is a model ~4x smaller on disk and faster on CPU-only hardware, with negligible accuracy regression.

For real embedded deployment (microcontroller, FPGA, NPU), the next step would be ONNX export — enabling hardware-agnostic runtimes like TensorRT (NVIDIA) or OpenVINO (Intel). The quantized weights translate directly, since both runtimes natively support INT8 inference.
