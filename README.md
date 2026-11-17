# EdgeLentils

**Monocular driving-scene perception, from model training toward efficient deployment.**

EdgeLentils is an ongoing computer vision project exploring the accuracy, geometric assumptions, and runtime trade-offs of a lightweight camera-based Advanced Driver Assistance System (ADAS). It combines a fine-tuned YOLOv8n detector, calibration-based distance estimation, classical lane detection, and interpretable warning rules on KITTI driving images.

The goal is to carry the same perception pipeline from Python experimentation into a measured C++/TensorRT implementation on constrained GPU hardware. The current repository contains the Python pipeline, training configurations, evaluation reports, unit tests, and a GitHub Actions workflow. Model export, C++ deployment, and runtime benchmarking are the next stages; real-time performance has not yet been established.

## Current capabilities

| Component | Implemented behavior |
|---|---|
| Data preparation | KITTI-to-YOLO label conversion, class mapping, calibration parsing, and two train/validation splits |
| Object detection | YOLOv8n baseline training and a higher-resolution rectangular fine-tune for Car, Pedestrian, and Cyclist |
| Monocular geometry | Forward depth and lateral position from box ground-contact points, camera intrinsics, and a flat-road model with a small roll correction |
| Lane detection | Perspective warp, color/gradient thresholding, scored boundary candidates, polynomial fitting, and geometric plausibility checks |
| Warning logic | Forward proximity warnings within a fixed ego-path corridor and instantaneous lane-center offset warnings |
| Visual inspection | A single-image demo combining detections, estimated positions, lane boundaries, corridor overlays, and warning states |
| Verification | pytest coverage for data conversion, calibration, geometry, lane fitting, warning rules, and training configuration; Ruff and pytest in CI |

The current data flow is:

```text
Image + camera calibration
  |
  +-- YOLOv8n --> boxes --> depth + lateral position --> ego-corridor proximity rule
  |
  +-- Classical lane detection --> lane-center offset --> departure-offset rule
  |
  +-- Combined overlay + warning state
```

The forward-warning corridor is independent of lane detection, so a missing lane fit does not disable the proximity check. This is a research and engineering prototype: its warning rules are simplified, and it does not implement vehicle control.

## Experiments and results

### Detection

Reported results use the Chen/3DOP split of KITTI's labeled object-detection training set: **3,712 training images and 3,769 validation images**. This split separates driving sequences to reduce temporal leakage. A seeded, class-stratified random split is also generated, but its training/evaluation comparison remains planned.

The label mapping merges `Van` and `Truck` into `Car`, and `Person_sitting` into `Pedestrian`; `Tram`, `Misc`, and `DontCare` annotations are dropped. The numbers below are Ultralytics validation metrics on these mapped classes, rather than official KITTI difficulty-specific benchmark scores.

| Validation metric | Square-640 baseline | Rectangular-1024 fine-tune |
|---|---:|---:|
| Overall mAP@0.50 | 0.569 | **0.629** |
| Overall mAP@0.50:0.95 | 0.340 | **0.387** |
| Car AP@0.50 | 0.875 | **0.902** |
| Pedestrian AP@0.50 | 0.517 | **0.564** |
| Cyclist AP@0.50 | 0.315 | **0.421** |

The primary model continues from the baseline's best checkpoint, using rectangular training at `imgsz=1024`, no mosaic augmentation, and a reduced learning rate. It improves overall mAP@0.50 by **0.060**, with the largest class gain for Cyclist (**+0.106 AP@0.50**). This is a comparison of two training stages with several configuration changes, so it does not isolate resolution as the sole cause of improvement. Pedestrian and Cyclist performance still trails Car; class imbalance and small-object representation remain priorities for further experiments.

Exact scores and checkpoint paths are recorded in [metrics.json](metrics.json) and [metrics_baseline_square640.json](metrics_baseline_square640.json). The training setups are in [configs/](configs/).

### Distance estimation

The geometric model estimates forward depth from the bottom-center of an object's bounding box. Without roll correction, the model is `Z = fy * H / (v_bottom - cy)`, where `H` is the assumed camera height (1.65 m). Lateral position follows from the same pinhole model. The implementation applies a small camera-roll correction using both image coordinates.

Evaluation uses **ground-truth 2D boxes** and compares estimated depth against KITTI's annotated 3D forward position. It therefore measures geometry error independently of detector localization and recall; it is not an end-to-end distance score for predicted boxes.

| Ground-truth depth | Objects evaluated | Median absolute error | 95th-percentile absolute error |
|---|---:|---:|---:|
| <10 m | 3,338 | 1.15 m | 3.49 m |
| 10–25 m | 7,428 | 2.21 m | 7.11 m |
| 25–50 m | 6,882 | 5.82 m | 30.53 m |
| 50 m and above | 2,243 | 11.97 m | 86.57 m |
| **Overall** | **19,891** | **2.90 m** | **23.56 m** |

Overall mean absolute error is **11.34 m**, reflecting a substantial long-range error tail as ground-contact points approach the horizon. The report retains 32 flagged near-horizon cases and records exclusions for 4 objects behind the camera and 52 with invalid ground-plane projections.

The **−1.4° roll parameter was fitted on this same validation split**. These are calibration-development results, not an independent held-out estimate of generalization. Full statistics, including per-class errors, are available in [distance_validation.json](distance_validation.json).

### Lanes and warnings

Lane detection has been assessed through visual inspection and synthetic unit tests; **no quantitative lane-accuracy score is reported**. Candidate scoring improves the search beyond simply choosing the strongest histogram peaks, but geometric plausibility alone cannot reliably distinguish painted boundaries from curbs, fences, or parked-car edges. On streets without lane markings, the pipeline can still return a plausible but incorrect fit.

The current rules are deliberately inspectable:

- **Forward Collision Warning (FCWS):** estimated depth below 15 m and absolute lateral offset below 1.41 m. This is a fixed proximity rule; it does not estimate closing speed or time to collision. The corridor assumes straight vehicle motion.
- **Lane Departure Warning (LDWS):** absolute estimated lane-center offset above 0.3 m. This is an instantaneous offset check, without temporal drift tracking. If no lane fit is available, no departure warning is emitted.

These thresholds are prototype choices, not validated operating limits. The KITTI object-detection images used here are independently sampled stills, so sequential driving behavior is outside the current evaluation.

## Reproducing the pipeline

### Environment

Development and training use **Python 3.10 on Windows 11**, with an **NVIDIA GTX 1650, 4 GB VRAM**. Batch size and worker counts in the configurations were selected using local memory probes. The GPU has no Tensor Cores; future FP16 runtime benefits will need to be measured on this hardware.

The recorded Python stack includes PyTorch 2.5.1, torchvision 0.20.1, Ultralytics 8.3.143, and OpenCV 4.11.0.86. See [requirements.txt](requirements.txt) and the [environment notes](docs/environment.md) for the pinned setup. CUDA/cuDNN/TensorRT deployment versions will be finalized during export work.

From the repository root, create an environment and install the recorded CUDA 12.1 PyTorch build before the remaining dependencies. This requires a compatible NVIDIA driver:

```powershell
conda create -n edgelentils python=3.10
conda activate edgelentils
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

### Prepare KITTI

Obtain the left color images, labels, and calibration files for the [KITTI 2D object-detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php) and extract the training files into:

```text
data/raw/training/
  image_2/    # 000000.png, ...
  label_2/    # 000000.txt, ...
  calib/      # 000000.txt, ...
```

Then run these commands in order:

```powershell
python -m perception.make_splits
python -m perception.convert_labels
```

The split script downloads the published Chen split lists and also creates a seeded random split. Conversion generates YOLO labels and dataset YAML files, and links `data/processed/images/` to the raw images using a Windows junction or a symlink. Generated split files contain absolute paths; regenerate them after moving the repository.

### Train and evaluate

Run the baseline first, then fine-tune its best checkpoint:

```powershell
python -m perception.train --config configs/train_baseline.yaml
python -m perception.train --config configs/finetune_rect1024.yaml
```

The baseline initializes from `yolov8n.pt`. The fine-tune expects `runs/detect/baseline/weights/best.pt`; its checkpoint is written under `runs/detect/rect1024_finetune/`. Both configurations target GPU device `0` and set a seed and deterministic training options.

To evaluate an available fine-tuned checkpoint and regenerate the geometry report:

```powershell
python -m perception.eval --weights runs/detect/rect1024_finetune/weights/best.pt --data data/processed/chen.yaml --out metrics_reproduced.json
python -m perception.distance --split val --out distance_validation_reproduced.json
```

The detection evaluator calls Ultralytics validation with its default inference settings; the training resolutions above describe the training configurations, not an explicitly controlled runtime-resolution benchmark.

### Render a frame

After training, run the combined demo on a KITTI image:

```powershell
python -m perception.demo --image data/processed/images/000010.png --weights runs/detect/rect1024_finetune/weights/best.pt --out overlay.png
```

Calibration defaults to the matching file in `data/raw/training/calib/`; `--calib` supplies an explicit path. The output overlays estimated depth/lateral position, matched ground-truth depth when available, lane boundaries, the straight ego corridor, and warning states. Lane detection can also be inspected without a detector checkpoint:

```powershell
python -m perception.lanes --image data/raw/training/image_2/000010.png --out lanes_overlay.png
```

**Data, trained weights, and generated demo outputs are excluded from version control.** The committed JSON reports document completed local experiments; reproducing detector results requires obtaining KITTI and training the checkpoints.

### Tests and CI

For CPU-only checks, install the lightweight CI dependencies in a separate Python 3.10 environment:

```powershell
python -m pip install -r requirements-ci.txt
python -m ruff check .
python -m pytest tests/ -v
```

The [GitHub Actions workflow](.github/workflows/ci.yml) runs lint and tests on pushes and pull requests. Tests requiring local KITTI files skip when those files are absent. GPU training, detector evaluation, and the combined demo are outside the current CI coverage.

## Planned work

The next milestones complete the deployment and measurement side of the project:

1. **Export and validate:** export PyTorch weights to ONNX, check numerical agreement, build TensorRT FP32/FP16 engines, and verify detection accuracy after each conversion.
2. **Implement C++ inference:** build a CMake/TensorRT/CUDA/OpenCV pipeline with preprocessing, decoding/NMS, geometry and warning logic, and video input/output.
3. **Benchmark across runtimes:** compare PyTorch GPU, ONNX Runtime, TensorRT FP32, and TensorRT FP16 on matched inputs, reporting latency, throughput, and accuracy parity on the GTX 1650.
4. **Package and document:** add training/export and deployment Docker images, a C++ smoke test, a results write-up, and a recorded video demonstration.

Further experiments are intentionally separate from those core milestones:

- Quantify the effect of sequence-aware versus random splitting on detection metrics.
- Evaluate class-imbalance mitigation and targeted augmentation, including offline rectangular mosaic, before considering larger detectors and their runtime costs.
- Explore learned lane detection to address the classical pipeline's remaining ambiguity and lighting sensitivity.
- Use sequential driving data for temporal lane-departure logic, and heading measurements for a curved ego-path corridor.
- Extend benchmarking to Tensor Core hardware or Jetson, then investigate INT8 deployment; multi-camera or LiDAR fusion would be a later extension.

The detailed scope and development history are in the [project plan](EdgeADAS_Project_Plan.md) and [learning log](docs/EdgeADAS_Learning_Log.md). This README reflects the implemented code and committed reports; the plan also retains the original schedule and historical progress notes.
