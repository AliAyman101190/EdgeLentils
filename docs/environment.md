# Environment

Pinned during Phase 1 setup, per `EdgeADAS_Project_Plan.md` Section 5. Do not bump
any of these without re-checking compatibility against the installed driver.

## Hardware

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GB VRAM, Turing — no Tensor Cores |
| Driver | 592.27 (`nvidia-smi`), CUDA 13.1 ceiling |
| OS | Windows 11 Home 10.0.26200 |

## Python environment

Conda env `yolov8-gpu`, Python 3.10.16. Exact versions in [`requirements.txt`](../requirements.txt).

| Package | Version | Notes |
|---|---|---|
| torch | 2.5.1 | built for cu121, cuDNN 9 (conda `pytorch` channel build `py3.10_cuda12.1_cudnn9_0`) |
| torchvision | 0.20.1 | |
| ultralytics | 8.3.143 | |
| opencv-python | 4.11.0.86 | |
| numpy | 1.23.5 | |

Driver 592.27 supports up to CUDA 13.1; the installed PyTorch build targets CUDA
12.1, which is within the driver's backward-compatible range — no reinstall needed.

## TBD — Phase 4 (export)

cuDNN and TensorRT versions are not pinned yet. Installing TensorRT now would mean
guessing at a version before the ONNX export work defines what's actually needed;
this gets pinned in Phase 4 against whatever ONNX opset the export step settles on.
