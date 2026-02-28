# CRAN-AI (TorchGPU)

A 6G-ready AI-native Cognitive Relay Framework built from scratch (DF/CF, robust imperfect CSI, multi-objective optimization, hybrid DF/CF selection, system-parameter generalization, and deployment/real-time evidence).

## Phase (0): Environment bootstrap

### 0) Check NVIDIA driver
```bash
nvidia-smi
```

### 1) Create the pinned conda environment
```bash
conda env create -f environment.yml
conda activate TorchGPU
```

### 2) Register Jupyter kernel (optional)
```bash
python -m ipykernel install --user --name TorchGPU --display-name "TorchGPU"
```

### 3) Verify Python + CUDA inside PyTorch
```bash
python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("ok: matmul on", y.device, "shape=", tuple(y.shape))
else:
    raise SystemExit("CUDA not available inside PyTorch. Check driver/conda env.")
PY
```

## Notes
- This repo is designed for **GPU-first** execution. During training, all tensors and computations must stay on CUDA (no NumPy ops in the training path).
- Next phases will add the `cran/` package, configs, tests, experiments, and visualization modules.
