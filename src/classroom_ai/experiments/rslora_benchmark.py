"""Paired Rank-Stabilized LoRA and full-finetune benchmark helpers.

This module measures training-time GPU memory and computes paired validation
trade-offs. It does not infer VRAM savings from parameter counts or model names.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RsLoRASettings:
    rank: int = 4
    alpha: int = 8
    min_layer_index: int = 13
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("rank and alpha must be positive")
        if self.min_layer_index < 0:
            raise ValueError("min_layer_index must be non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def select_conv_targets(model: Any, min_layer_index: int = 13) -> list[str]:
    """Select regular Conv2d layers in the YOLO neck/head, excluding depthwise ops."""
    import torch.nn as nn

    targets: list[str] = []
    for name, module in model.named_modules():
        match = re.match(r"^model\.(\d+)(?:\.|$)", name)
        if (
            match
            and int(match.group(1)) >= min_layer_index
            and isinstance(module, nn.Conv2d)
            and module.groups == 1
        ):
            targets.append(name)
    if not targets:
        raise ValueError(
            f"No Conv2d target found at or after model layer {min_layer_index}"
        )
    return targets


def make_rslora_detection_trainer(settings: RsLoRASettings = RsLoRASettings()):
    """Create an Ultralytics DetectionTrainer that injects and freezes rsLoRA adapters."""
    from peft import LoraConfig, inject_adapter_in_model
    from ultralytics.models.yolo.detect import DetectionTrainer

    class RsLoRADetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            targets = select_conv_targets(model, settings.min_layer_index)
            config = LoraConfig(
                r=settings.rank,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                target_modules=targets,
                bias="none",
                use_rslora=True,
            )
            model = inject_adapter_in_model(config, model)

            trainable = 0
            total = 0
            for name, parameter in model.named_parameters():
                total += parameter.numel()
                is_adapter = ".lora_A." in name or ".lora_B." in name
                parameter.requires_grad_(is_adapter)
                if is_adapter:
                    trainable += parameter.numel()
            if trainable == 0:
                raise RuntimeError("rsLoRA injection created no trainable parameters")

            self.rslora_benchmark = {
                "method": "rsLoRA",
                "rank": settings.rank,
                "alpha": settings.alpha,
                "scaling": settings.alpha / (settings.rank**0.5),
                "target_min_layer_index": settings.min_layer_index,
                "target_conv2d_count": len(targets),
                "target_modules": targets,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_fraction": trainable / total,
            }
            return model

    return RsLoRADetectionTrainer


class PeakVramMonitor:
    """Sample per-GPU NVML memory while a training call is running.

    The reported peak is process-independent device memory usage sampled at the
    configured interval. Peak increment is relative to device usage at start.
    """

    def __init__(self, device_count: int = 2, interval_s: float = 0.05):
        if device_count < 1 or interval_s <= 0:
            raise ValueError("device_count and interval_s must be positive")
        self.device_count = device_count
        self.interval_s = interval_s
        self._nvml = None
        self._handles = []
        self._baseline: list[int] = []
        self._peak: list[int] = []
        self._names: list[str] = []
        self._totals: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0

    def _sample(self) -> list[int]:
        values = [
            int(self._nvml.nvmlDeviceGetMemoryInfo(handle).used)
            for handle in self._handles
        ]
        self._samples += 1
        self._peak = [max(old, new) for old, new in zip(self._peak, values)]
        return values

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                self._stop.set()
                return
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakVramMonitor:
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError(
                "NVML Python bindings are required (install nvidia-ml-py)"
            ) from exc
        self._nvml = pynvml
        self._nvml.nvmlInit()
        count = self._nvml.nvmlDeviceGetCount()
        if count < self.device_count:
            self._nvml.nvmlShutdown()
            raise RuntimeError(
                f"Expected {self.device_count} visible GPUs, found {count}"
            )
        self._handles = [
            self._nvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(self.device_count)
        ]
        self._names = []
        self._totals = []
        for handle in self._handles:
            name = self._nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            self._names.append(str(name))
            self._totals.append(
                int(self._nvml.nvmlDeviceGetMemoryInfo(handle).total)
            )
        self._baseline = self._sample()
        self._peak = list(self._baseline)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        if self._nvml is not None:
            try:
                self._sample()
            finally:
                self._nvml.nvmlShutdown()

    @staticmethod
    def _mib(value: int) -> float:
        return round(value / (1024**2), 2)

    def result(self) -> dict[str, Any]:
        if not self._handles or len(self._peak) != self.device_count:
            raise RuntimeError("VRAM monitor has no completed samples")
        devices = []
        for index, name in enumerate(self._names):
            delta = max(0, self._peak[index] - self._baseline[index])
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "total_mib": self._mib(self._totals[index]),
                    "idle_used_mib": self._mib(self._baseline[index]),
                    "peak_used_mib": self._mib(self._peak[index]),
                    "peak_increment_mib": self._mib(delta),
                }
            )
        return {
            "measurement": "NVML sampled device memory during training only",
            "sampling_interval_ms": round(self.interval_s * 1000, 1),
            "sample_count": self._samples,
            "devices": devices,
            "peak_increment_max_device_mib": max(
                device["peak_increment_mib"] for device in devices
            ),
            "peak_increment_sum_devices_mib": round(
                sum(device["peak_increment_mib"] for device in devices), 2
            ),
        }


def parameter_counts(model: Any) -> dict[str, int | float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def relative_percent_change(reference: float, candidate: float) -> float | None:
    """Return candidate's relative change from reference in percent."""
    if reference == 0:
        return None
    return round((candidate - reference) / abs(reference) * 100, 4)


def compare_runs(
    full_metrics: dict[str, float],
    adapter_metrics: dict[str, float],
    full_vram: dict[str, Any],
    adapter_vram: dict[str, Any],
) -> dict[str, Any]:
    """Compute empirical memory savings and validation mAP trade-offs."""
    full_peak = float(full_vram["peak_increment_max_device_mib"])
    adapter_peak = float(adapter_vram["peak_increment_max_device_mib"])
    full_map = float(full_metrics["map50_95"])
    adapter_map = float(adapter_metrics["map50_95"])
    return {
        "primary_metric": "mAP50-95 on the locked validation split",
        "vram": {
            "full_peak_increment_max_device_mib": full_peak,
            "rslora_peak_increment_max_device_mib": adapter_peak,
            "saving_percent": (
                round((full_peak - adapter_peak) / full_peak * 100, 4)
                if full_peak > 0
                else None
            ),
            "full_peak_increment_sum_devices_mib": full_vram[
                "peak_increment_sum_devices_mib"
            ],
            "rslora_peak_increment_sum_devices_mib": adapter_vram[
                "peak_increment_sum_devices_mib"
            ],
        },
        "accuracy_tradeoff": {
            "full_map50_95": full_map,
            "rslora_map50_95": adapter_map,
            "absolute_difference": round(adapter_map - full_map, 6),
            "absolute_difference_percentage_points": round(
                (adapter_map - full_map) * 100, 4
            ),
            "relative_change_percent": relative_percent_change(full_map, adapter_map),
            "relative_loss_percent": (
                round((full_map - adapter_map) / abs(full_map) * 100, 4)
                if full_map != 0
                else None
            ),
        },
        "interpretation": (
            "Positive relative_loss_percent means rsLoRA mAP50-95 is lower; "
            "negative means it is higher. This is validation mAP, not classification accuracy."
        ),
    }
