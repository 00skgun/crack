#!/usr/bin/env python3
"""Export the ESW DeepLabV3 checkpoint to a Raspberry Pi friendly NCNN model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np
import pnnx
import torch
import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class OutputOnly(nn.Module):
    """Expose only the segmentation tensor instead of torchvision's dict output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state_dict mapping")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    return state_dict


def build_model(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    aux_loss = any(key.startswith("aux_classifier.") for key in state_dict)
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=None,
        weights_backbone=None,
        aux_loss=aux_loss,
    )
    model.classifier = DeepLabHead(2048, 1)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return OutputOnly(model).eval()


def run_ncnn(
    net: ncnn.Net,
    input_chw: np.ndarray,
    threads: int,
) -> np.ndarray:
    for _ in range(3):
        extractor = net.create_extractor()
        if hasattr(extractor, "set_num_threads"):
            extractor.set_num_threads(threads)
        input_result = extractor.input("in0", ncnn.Mat(input_chw))
        if input_result != 0:
            raise RuntimeError(f"NCNN input failed with code {input_result}")
        output_result, output = extractor.extract("out0")
        if output_result != 0:
            raise RuntimeError(f"NCNN extraction failed with code {output_result}")
        output_array = np.asarray(output).squeeze()
        if np.isfinite(output_array).all():
            return output_array
    raise RuntimeError("NCNN produced non-finite output in three consecutive attempts")


def minmax_255(array: np.ndarray) -> np.ndarray:
    value_range = float(array.max() - array.min())
    if value_range == 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - array.min()) * (255.0 / value_range)).astype(np.float32)


def comparison_metrics(reference: np.ndarray, converted: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(reference - converted)
    reference_normalized = minmax_255(reference)
    converted_normalized = minmax_255(converted)
    normalized_error = np.abs(reference_normalized - converted_normalized)
    return {
        "raw_max_abs_error": float(absolute_error.max()),
        "raw_mean_abs_error": float(absolute_error.mean()),
        "normalized_0_255_max_abs_error": float(normalized_error.max()),
        "normalized_0_255_mean_abs_error": float(normalized_error.mean()),
        "threshold_127_pixel_disagreement_ratio": float(
            np.mean((reference_normalized >= 127.0) != (converted_normalized >= 127.0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--validation-dir", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)

    state_dict = load_state_dict(checkpoint_path)
    model = build_model(state_dict)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    buffer_count = sum(buffer.numel() for buffer in model.buffers())

    sample = torch.rand(1, 3, args.img_size, args.img_size, dtype=torch.float32)
    with torch.inference_mode():
        reference = model(sample).cpu().numpy().squeeze()

    stem = f"esw_seg_model_fp16_{args.img_size}"
    ncnn_param = output_dir / f"{stem}.ncnn.param"
    ncnn_bin = output_dir / f"{stem}.ncnn.bin"

    with tempfile.TemporaryDirectory(prefix="esw-pnnx-") as temporary_dir:
        work_dir = Path(temporary_dir)
        pnnx.export(
            model,
            (work_dir / f"{stem}.torchscript.pt").as_posix(),
            (sample,),
            device="cpu",
            optlevel=2,
            fp16=True,
            pnnxparam=(work_dir / f"{stem}.pnnx.param").as_posix(),
            pnnxbin=(work_dir / f"{stem}.pnnx.bin").as_posix(),
            pnnxpy=(work_dir / f"{stem}_pnnx.py").as_posix(),
            pnnxonnx=(work_dir / f"{stem}.pnnx.onnx").as_posix(),
            ncnnparam=ncnn_param.as_posix(),
            ncnnbin=ncnn_bin.as_posix(),
            ncnnpy=(work_dir / f"{stem}_ncnn.py").as_posix(),
            check_trace=True,
        )

    net = ncnn.Net()
    net.opt.num_threads = args.threads
    net.opt.use_packing_layout = True
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    # Cortex-A72 has no native ARMv8.2 FP16 arithmetic; keep accumulation in FP32.
    net.opt.use_fp16_arithmetic = False
    if net.load_param(str(ncnn_param)) != 0:
        raise RuntimeError("Failed to load generated NCNN param")
    if net.load_model(str(ncnn_bin)) != 0:
        raise RuntimeError("Failed to load generated NCNN weights")

    input_chw = np.ascontiguousarray(sample.numpy()[0])
    converted = run_ncnn(net, input_chw, args.threads)
    if converted.shape != reference.shape:
        raise RuntimeError(
            f"Output shape mismatch: PyTorch={reference.shape}, NCNN={converted.shape}"
        )

    random_input_metrics = comparison_metrics(reference, converted)
    max_abs_error = random_input_metrics["raw_max_abs_error"]
    if not np.isfinite(converted).all() or max_abs_error > 0.05:
        raise RuntimeError(f"NCNN numerical validation failed: max_abs={max_abs_error}")

    sample_validation: list[dict[str, float | str]] = []
    if args.validation_dir is not None:
        validation_dir = args.validation_dir.resolve()
        image_paths = sorted(
            path
            for path in validation_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (args.img_size, args.img_size))
            input_chw_image = np.ascontiguousarray(
                image_rgb.transpose(2, 0, 1),
                dtype=np.float32,
            ) / 255.0
            with torch.inference_mode():
                torch_image_output = (
                    model(torch.from_numpy(input_chw_image).unsqueeze(0))
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            ncnn_image_output = run_ncnn(net, input_chw_image, args.threads)
            image_metrics = comparison_metrics(torch_image_output, ncnn_image_output)
            image_metrics["file"] = image_path.name
            sample_validation.append(image_metrics)

    run_times_ms: list[float] = []
    for _ in range(max(0, args.benchmark_runs)):
        started = time.perf_counter()
        run_ncnn(net, input_chw, args.threads)
        run_times_ms.append((time.perf_counter() - started) * 1000.0)

    metadata = {
        "architecture": "torchvision.deeplabv3_resnet101",
        "output_channels": 1,
        "input_shape": [1, 3, args.img_size, args.img_size],
        "output_shape": list(reference.shape),
        "parameter_count": parameter_count,
        "buffer_count": buffer_count,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_bytes": checkpoint_path.stat().st_size,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "ncnn_param": ncnn_param.name,
        "ncnn_bin": ncnn_bin.name,
        "ncnn_param_bytes": ncnn_param.stat().st_size,
        "ncnn_bin_bytes": ncnn_bin.stat().st_size,
        "ncnn_param_sha256": sha256_file(ncnn_param),
        "ncnn_bin_sha256": sha256_file(ncnn_bin),
        "weight_storage": "fp16",
        "runtime_accumulation": "fp32",
        "pnnx_optlevel": 2,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "pnnx_version": getattr(pnnx, "__version__", "unknown"),
        "pytorch_output_min": float(reference.min()),
        "pytorch_output_max": float(reference.max()),
        "ncnn_output_min": float(converted.min()),
        "ncnn_output_max": float(converted.max()),
        "ncnn_max_abs_error": max_abs_error,
        "ncnn_mean_abs_error": random_input_metrics["raw_mean_abs_error"],
        "normalized_0_255_max_abs_error": random_input_metrics[
            "normalized_0_255_max_abs_error"
        ],
        "normalized_0_255_mean_abs_error": random_input_metrics[
            "normalized_0_255_mean_abs_error"
        ],
        "threshold_127_pixel_disagreement_ratio": random_input_metrics[
            "threshold_127_pixel_disagreement_ratio"
        ],
        "sample_validation": sample_validation,
        "benchmark_host": "conversion host; not Raspberry Pi 4",
        "benchmark_threads": args.threads,
        "benchmark_runs_ms": run_times_ms,
        "benchmark_median_ms": statistics.median(run_times_ms) if run_times_ms else None,
    }
    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
