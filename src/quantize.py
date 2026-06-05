import argparse
import os
import time
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import mlflow

from model import build_model


def export_onnx(checkpoint_path: str, onnx_path: str):
    device = torch.device("cpu")
    model = build_model(num_classes=10, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"FP32 ONNX exported → {onnx_path}")


def measure_latency(onnx_path: str, n_runs: int = 100) -> float:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(10):
        session.run(None, {name: dummy})
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {name: dummy})
    return (time.perf_counter() - start) / n_runs * 1000  # ms


def quantize(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_fp32)), exist_ok=True)

    print("Step 1/3 — exporting FP32 ONNX...")
    export_onnx(args.checkpoint, args.onnx_fp32)

    print("Step 2/3 — quantizing to INT8...")
    quantize_dynamic(args.onnx_fp32, args.onnx_int8, weight_type=QuantType.QUInt8)
    print(f"INT8 ONNX saved → {args.onnx_int8}")

    print("Step 3/3 — benchmarking...")
    size_fp32    = os.path.getsize(args.onnx_fp32) / (1024 ** 2)
    size_int8    = os.path.getsize(args.onnx_int8) / (1024 ** 2)
    latency_fp32 = measure_latency(args.onnx_fp32)
    latency_int8 = measure_latency(args.onnx_int8)

    print(f"\n{'='*45}")
    print(f"  Model size  FP32 : {size_fp32:.2f} MB")
    print(f"  Model size  INT8 : {size_int8:.2f} MB  ({size_int8/size_fp32*100:.1f}%)")
    print(f"  Latency     FP32 : {latency_fp32:.2f} ms/sample")
    print(f"  Latency     INT8 : {latency_int8:.2f} ms/sample  ({latency_int8/latency_fp32*100:.1f}%)")
    print(f"{'='*45}")

    if args.run_id:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metrics({
                "size_fp32_mb":    size_fp32,
                "size_int8_mb":    size_int8,
                "latency_fp32_ms": latency_fp32,
                "latency_int8_ms": latency_int8,
            })
            mlflow.log_artifact(args.onnx_int8)
            print(f"Logged to MLflow run {args.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../models/mobilenet_finetuned.pth")
    parser.add_argument("--onnx_fp32",  default="../models/mobilenet_fp32.onnx")
    parser.add_argument("--onnx_int8",  default="../models/mobilenet_int8.onnx")
    parser.add_argument("--run_id",     default=None)
    args = parser.parse_args()
    quantize(args)
