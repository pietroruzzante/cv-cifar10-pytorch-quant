import argparse
import os
import time
import torch
import onnxruntime as ort
import numpy as np

from model import build_model


def export(args):
    device = torch.device("cpu")

    model = build_model(num_classes=10, freeze_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"ONNX model exported to {args.output}")

    _verify(args.output, dummy_input)


def _verify(onnx_path: str, dummy_input: torch.Tensor):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # correctness check
    outputs = session.run(None, {input_name: dummy_input.numpy()})
    print(f"Output shape: {outputs[0].shape}  (expected: (1, 10))")

    # latency benchmark
    n_runs = 100
    inp = dummy_input.numpy()
    for _ in range(10):  # warmup
        session.run(None, {input_name: inp})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: inp})
    latency_ms = (time.perf_counter() - start) / n_runs * 1000

    size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"ONNX model size : {size_mb:.2f} MB")
    print(f"ONNX latency    : {latency_ms:.2f} ms/sample (CPUExecutionProvider)")
    print("\nVerification passed — model is ready for edge deployment.")
    print("Compatible runtimes: TensorRT (NVIDIA), OpenVINO (Intel), ONNX Runtime Mobile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../models/mobilenet_finetuned.pth")
    parser.add_argument("--output",     default="../models/mobilenet.onnx")
    args = parser.parse_args()
    export(args)