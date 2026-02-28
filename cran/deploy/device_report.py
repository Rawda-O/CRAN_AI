from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class DeviceReport:
    python: str
    platform: str
    torch: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[int]
    gpu_name: Optional[str]
    gpu_count: int


def build_device_report() -> Dict:
    cuda_av = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_av else None
    gpu_count = torch.cuda.device_count() if cuda_av else 0

    rep = DeviceReport(
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        torch=torch.__version__,
        cuda_available=cuda_av,
        cuda_version=torch.version.cuda if hasattr(torch.version, "cuda") else None,
        cudnn_version=torch.backends.cudnn.version() if cuda_av else None,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
    )
    return asdict(rep)
