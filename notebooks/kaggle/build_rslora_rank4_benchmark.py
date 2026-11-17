"""Build the private Kaggle notebook for the paired rsLoRA Rank-4 pilot."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "classroom_ai" / "experiments" / "rslora_benchmark.py"
OUTPUT = Path(__file__).with_name("rslora_rank4_benchmark_20261109.ipynb")


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    module_source = MODULE.read_text(encoding="utf-8")
    cells = [
        markdown(
            """# Benchmark rsLoRA Rank 4 — VRAM so với mAP

Notebook Private. So sánh full fine-tune với rsLoRA rank 4 trên cùng checkpoint, cùng dữ liệu, seed, lịch huấn luyện và hai GPU T4.

Split theo video đã khóa: 477 ảnh train từ Thanh.MOV và 330 ảnh validation từ Thanh (1).MOV. Chỉ dùng mAP50–95 làm chỉ số chất lượng chính; P/R và mAP50 là phụ. Peak VRAM lấy mẫu thực bằng NVML trong lệnh train. Chưa có video test độc lập, vì vậy kết luận chỉ là pilot validation.

Checkpoint đầu vào có tên rsLoRA nhưng không có adapter Rank/alpha được xác nhận. Nhánh bên dưới tạo mới rsLoRA rank 4, alpha 8, scaling alpha/sqrt(rank), trên các convolution regular ở YOLO26 neck/head từ layer 13 trở đi. """
        ),
        code(
            """import importlib.metadata
import importlib.util
import os
import subprocess
import sys

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 2, (
    "Chọn GPU T4 x2 trong Settings; notebook không tự đổi accelerator."
)
gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
assert all("T4" in name.upper() for name in gpu_names), gpu_names
print("GPU:", gpu_names)

def ensure_package(distribution, module_name, pinned_version):
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != pinned_version or importlib.util.find_spec(module_name) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"{distribution}=={pinned_version}"]
        )

ensure_package("ultralytics", "ultralytics", "8.4.183")
ensure_package("peft", "peft", "0.17.1")
ensure_package("nvidia-ml-py", "pynvml", "12.570.86")

from ultralytics import YOLO, __version__, settings
settings.update(
    {
        "sync": False,
        "wandb": False,
        "comet": False,
        "clearml": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
    }
)
assert __version__ == "8.4.183", __version__
print("Ultralytics:", __version__)
print("PEFT:", importlib.metadata.version("peft"))
"""),
        code(
            "%%writefile /kaggle/working/rslora_benchmark.py\n" + module_source
        ),
        code(
            """from pathlib import Path
import datetime
import hashlib
import json
import shutil
import tempfile
import time
import uuid

import yaml
from ultralytics import YOLO

from rslora_benchmark import (
    PeakVramMonitor,
    RsLoRASettings,
    compare_runs,
    make_rslora_detection_trainer,
    parameter_counts,
)

INPUT = Path("/kaggle/input")

def unique_file(name):
    found = {path.resolve() for path in INPUT.rglob(name) if path.is_file()}
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name}; found {len(found)}. "
            "Attach the existing Private Thanh Detect dataset."
        )
    return next(iter(found))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

bundle = unique_file("bundle_metadata.json").parent
metadata = json.loads((bundle / "bundle_metadata.json").read_text(encoding="utf-8"))
checkpoint = bundle / "weights" / "LIGHT_CONTRAST_rsLoRA_best_E1.pt"
source_data = bundle / "dataset"
manifest = json.loads(unique_file("frames_manifest.json").read_text(encoding="utf-8"))
source_report = json.loads(
    (source_data / "conversion_report.json").read_text(encoding="utf-8")
)
assert metadata["checkpoint_sha256"] == sha256(checkpoint), "Checkpoint hash mismatch"
assert len(manifest["frames"]) == 807
assert source_report["counts"] in (
    {"train": 330, "val": 477},
    {"train": 477, "val": 330},
)
assert source_report["boxes"] in (
    {"train": 1387, "val": 1799},
    {"train": 1799, "val": 1387},
)

desired_train_video = "941205c3ad5f6d300b399be5102d6191227b0b9dd7397e76732252a39a5c19a4"
desired_val_video = "b0a04470d591d46837fc81712cdbbfd74312444ae53e6f967131539a4b6770e5"
assert {
    source_report["train_video_sha256"],
    source_report["val_video_sha256"],
} == {desired_train_video, desired_val_video}
work = Path(tempfile.mkdtemp(prefix="rslora_rank4_"))
data_root = work / "dataset"
shutil.copytree(source_data, data_root)
split_swapped = source_report["train_video_sha256"] != desired_train_video
if split_swapped:
    for folder_name in ("images", "labels"):
        parent = data_root / folder_name
        (parent / "train").rename(parent / "_swap")
        (parent / "val").rename(parent / "train")
        (parent / "_swap").rename(parent / "val")

report = {
    "counts": {"train": 477, "val": 330},
    "boxes": {"train": 1799, "val": 1387},
    "train_video_sha256": desired_train_video,
    "val_video_sha256": desired_val_video,
}
manifest_rows = {row["filename"]: row for row in manifest["frames"]}
assert len(manifest_rows) == 807
seen = set()
for split, expected_count, expected_boxes, video_hash in (
    ("train", 477, 1799, desired_train_video),
    ("val", 330, 1387, desired_val_video),
):
    images = sorted((data_root / "images" / split).glob("*.jpg"))
    labels = sorted((data_root / "labels" / split).glob("*.txt"))
    assert len(images) == len(labels) == expected_count
    assert {image.stem for image in images} == {label.stem for label in labels}
    box_count = 0
    for image in images:
        row = manifest_rows[image.name]
        assert row["source_video_sha256"] == video_hash
        assert sha256(image) == row["image_sha256"]
        assert image.name not in seen
        seen.add(image.name)
        label_path = data_root / "labels" / split / (image.stem + ".txt")
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            assert len(fields) == 5 and fields[0] == "0"
            cx, cy, width, height = map(float, fields[1:])
            assert (
                0 <= cx - width / 2 <= cx + width / 2 <= 1
                and 0 <= cy - height / 2 <= cy + height / 2 <= 1
                and width > 0
                and height > 0
            )
            box_count += 1
    assert box_count == expected_boxes
    print(f"{split}: {expected_count} images, {box_count} person boxes, hashes verified")
assert len(seen) == 807

data_yaml = data_root / "data.yaml"
data_yaml.write_text(
    f"path: {data_root.as_posix()}\\n"
    "train: images/train\\n"
    "val: images/val\\n"
    "names:\\n"
    "  0: person\\n",
    encoding="utf-8",
)

run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_id += "_" + uuid.uuid4().hex[:8]
out = Path("/kaggle/working") / f"rslora_rank4_benchmark_{run_id}"
out.mkdir(parents=True, exist_ok=False)
protocol = {
    "schema_version": 1,
    "experiment": "paired YOLO26n full fine-tune vs rsLoRA rank 4",
    "source_checkpoint_sha256": sha256(checkpoint),
    "dataset": {
        "train_images": 477,
        "train_boxes": 1799,
        "train_video_sha256": desired_train_video,
        "validation_images": 330,
        "validation_boxes": 1387,
        "validation_video_sha256": desired_val_video,
        "independent_test_available": False,
        "input_split_swapped_in_working_copy": split_swapped,
    },
    "paired_training": {
        "epochs": 20,
        "imgsz": 1280,
        "batch": 16,
        "devices": gpu_names,
        "seed": 0,
        "deterministic": True,
        "optimizer": "auto",
        "workers": 2,
        "cache": False,
    },
    "rslora": {
        "rank": 4,
        "alpha": 8,
        "scaling": "alpha / sqrt(rank)",
        "target": "regular Conv2d in YOLO model layers 13 onward",
        "depthwise_convolutions_adapted": False,
    },
    "vram": {
        "method": "NVML sampled every 50 ms for each visible T4 during the train call",
        "primary_stat": "maximum per-device increase over idle at start of training call",
        "note": "The train call includes epoch validation; the peak is training-call peak.",
    },
    "primary_quality_metric": "mAP50-95 on the same held-out video validation split",
    "ultralytics_version": __version__,
    "created_utc": run_id,
}
(out / "benchmark_protocol.json").write_text(
    json.dumps(protocol, indent=2), encoding="utf-8"
)
print("Checkpoint SHA256:", protocol["source_checkpoint_sha256"])
print("Input dataset is read-only; reordered copies are under:", data_root)
print("Benchmark output:", out)
print("Locked protocol:", json.dumps(protocol, indent=2))
"""),
        code(
            """from pathlib import Path
import json
import time

full_name = "full_finetune"
full_dir = out / full_name
full_dir.mkdir(parents=True, exist_ok=True)
full_best = full_dir / "weights" / "best.pt"
full_record_path = out / "full_finetune.json"

if full_best.is_file() and full_record_path.is_file():
    full_record = json.loads(full_record_path.read_text(encoding="utf-8"))
    print("Reusing the completed full-finetune branch:", full_best)
else:
    full_model = YOLO(str(checkpoint))
    started = time.perf_counter()
    with PeakVramMonitor(device_count=2, interval_s=0.05) as memory:
        full_model.train(
            data=str(data_yaml),
            epochs=20,
            imgsz=1280,
            batch=16,
            device=[0, 1],
            seed=0,
            deterministic=True,
            optimizer="auto",
            workers=2,
            cache=False,
            pretrained=True,
            resume=False,
            project=str(out),
            name=full_name,
            exist_ok=True,
            plots=False,
            patience=20,
        )
    elapsed = time.perf_counter() - started
    full_best = Path(full_model.trainer.best)
    assert full_best.is_file(), f"Missing full-finetune best checkpoint: {full_best}"
    full_record = {
        "method": "full_finetune",
        "best_checkpoint": str(full_best),
        "best_checkpoint_sha256": sha256(full_best),
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram": memory.result(),
        "parameters": parameter_counts(full_model.trainer.model),
    }
    full_record_path.write_text(
        json.dumps(full_record, indent=2), encoding="utf-8"
    )
    del full_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(full_record, indent=2))
"""),
        code(
            """from pathlib import Path
import json
import time

rank4_name = "rslora_rank4"
rank4_dir = out / rank4_name
rank4_dir.mkdir(parents=True, exist_ok=True)
rank4_best = rank4_dir / "weights" / "best.pt"
rank4_record_path = out / "rslora_rank4.json"

if rank4_best.is_file() and rank4_record_path.is_file():
    rank4_record = json.loads(rank4_record_path.read_text(encoding="utf-8"))
    print("Reusing the completed rsLoRA branch:", rank4_best)
else:
    rank4_model = YOLO(str(checkpoint))
    trainer_class = make_rslora_detection_trainer(
        RsLoRASettings(rank=4, alpha=8, min_layer_index=13, dropout=0.0)
    )
    started = time.perf_counter()
    with PeakVramMonitor(device_count=2, interval_s=0.05) as memory:
        rank4_model.train(
            data=str(data_yaml),
            epochs=20,
            imgsz=1280,
            batch=16,
            device=[0, 1],
            seed=0,
            deterministic=True,
            optimizer="auto",
            workers=2,
            cache=False,
            pretrained=True,
            resume=False,
            trainer=trainer_class,
            project=str(out),
            name=rank4_name,
            exist_ok=True,
            plots=False,
            patience=20,
        )
    elapsed = time.perf_counter() - started
    rank4_best = Path(rank4_model.trainer.best)
    assert rank4_best.is_file(), f"Missing rsLoRA best checkpoint: {rank4_best}"
    rank4_record = {
        "method": "rsLoRA",
        "best_checkpoint": str(rank4_best),
        "best_checkpoint_sha256": sha256(rank4_best),
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram": memory.result(),
        "parameters": parameter_counts(rank4_model.trainer.model),
        "adapter": getattr(rank4_model.trainer, "rslora_benchmark", None),
    }
    assert rank4_record["adapter"] is not None, "Missing rsLoRA adapter metadata"
    rank4_record_path.write_text(
        json.dumps(rank4_record, indent=2), encoding="utf-8"
    )
    del rank4_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(rank4_record, indent=2))
"""),
        code(
            """def metric_values(metrics):
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

def validate_best(checkpoint_path, run_name):
    model = YOLO(str(checkpoint_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=1280,
        batch=8,
        device=0,
        classes=[0],
        conf=0.001,
        iou=0.7,
        plots=False,
        project=str(out),
        name=run_name,
        exist_ok=True,
    )
    return metric_values(metrics)

full_metrics_path = out / "full_validation_metrics.json"
rank4_metrics_path = out / "rslora_rank4_validation_metrics.json"
if full_metrics_path.is_file():
    full_metrics = json.loads(full_metrics_path.read_text(encoding="utf-8"))
else:
    full_metrics = validate_best(full_best, "validate_full_best")
    full_metrics_path.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
if rank4_metrics_path.is_file():
    rank4_metrics = json.loads(rank4_metrics_path.read_text(encoding="utf-8"))
else:
    rank4_metrics = validate_best(rank4_best, "validate_rslora_rank4_best")
    rank4_metrics_path.write_text(
        json.dumps(rank4_metrics, indent=2), encoding="utf-8"
    )

tradeoff = compare_runs(
    full_metrics,
    rank4_metrics,
    full_record["peak_vram"],
    rank4_record["peak_vram"],
)
comparison = {
    "schema_version": 1,
    "run_id": run_id,
    "protocol": protocol,
    "gpu_names": gpu_names,
    "ultralytics_version": __version__,
    "peft_version": importlib.metadata.version("peft"),
    "full_finetune": {
        **full_record,
        "validation_metrics": full_metrics,
    },
    "rslora_rank4": {
        **rank4_record,
        "validation_metrics": rank4_metrics,
    },
    "paired_comparison": tradeoff,
    "limitations": [
        "Pilot validation only; no independent video test set is available.",
        "One paired seed; no seed variance estimate.",
        "mAP50-95 is detection quality, not classification accuracy.",
        "NVML samples whole-device usage at 20 Hz during the train call.",
    ],
}
(out / "comparison.json").write_text(
    json.dumps(comparison, indent=2), encoding="utf-8"
)

vram = tradeoff["vram"]
quality = tradeoff["accuracy_tradeoff"]
report_text = f'''# rsLoRA Rank-4 paired benchmark

Status: completed pilot validation. No independent test video is available.

| Method | Peak training VRAM increase per GPU (MiB) | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Full fine-tune | {vram["full_peak_increment_max_device_mib"]:.2f} | {full_metrics["precision"]:.4f} | {full_metrics["recall"]:.4f} | {full_metrics["map50"]:.4f} | {full_metrics["map50_95"]:.4f} |
| rsLoRA rank 4 | {vram["rslora_peak_increment_max_device_mib"]:.2f} | {rank4_metrics["precision"]:.4f} | {rank4_metrics["recall"]:.4f} | {rank4_metrics["map50"]:.4f} | {rank4_metrics["map50_95"]:.4f} |

Measured per-GPU VRAM saving: {vram["saving_percent"]}%.

Relative mAP50-95 loss for rsLoRA versus full fine-tune: {quality["relative_loss_percent"]}%.
Absolute mAP50-95 difference: {quality["absolute_difference_percentage_points"]} percentage points.

The measurement samples NVML device memory during the train call at 50 ms intervals. Both methods use the same checkpoint, training split, validation split, seed, image size, global batch, epoch budget, optimizer setting, and two-T4 hardware. The full-finetune model is the reference; positive loss means rsLoRA scores lower. This single-video validation result is a pilot and does not establish generalization.
'''
(out / "report.md").write_text(report_text, encoding="utf-8")
print(report_text)
print("Artifacts:", sorted(path.name for path in out.iterdir()))
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
