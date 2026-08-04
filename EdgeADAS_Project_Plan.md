# EdgeADAS — Real-Time Perception & Deployment Pipeline

## Notes for Claude Code

This file is the working spec for this project. Treat it as the source of truth for scope.

- Work through **Section 2 (Deliverables Checklist)** top to bottom, checking off `- [ ]` → `- [x]` as items are completed. Don't skip ahead or reorder phases without flagging it — the phases are sequenced deliberately (each depends on the previous one working).
- **Section 3 ("Taking It Further") is explicitly out of scope for now.** Do not implement anything from it unless the user directly asks. It exists so that scope stays contained during the core build.
- **Section 4 (Timeline)** includes a learning block for each week — the user is new to several of these topics. Don't assume prior expertise in TensorRT, CUDA C++, or monocular geometry; budget explanation time accordingly when working through those weeks.
- **Section 5 (Hardware & Environment)** — versions marked TBD must be pinned during Week 1 setup based on the user's actual installed driver, not assumed. Always confirm CUDA/cuDNN/TensorRT compatibility before installing anything.
- When a session starts, check which week we're in and what's already checked off before proposing next steps.

---

## 1. Project Overview

### The idea

A lightweight Advanced Driver Assistance System (ADAS) that processes camera video and produces two real driving-relevant outputs:

- **Forward Collision Warning (FCWS):** detect vehicles/pedestrians/cyclists and estimate distance to each.
- **Lane Departure / Keeping Assist (LDWS/LKAS):** detect lane boundaries and flag departure from lane center.

The project has two acts, and the second is the actual differentiator:

1. **Train it** — a real object detection model + distance estimation + lane detection, evaluated with real metrics.
2. **Deploy it** — convert the trained model to run through TensorRT and serve it from a real-time C++ inference pipeline, with hard latency/FPS numbers proving it's fast enough to matter, not just accurate in a notebook.

### Scope (in scope for the 8-week core build)

- Single front-facing camera only (matches how FCWS/LDWS/LKAS actually work in production ADAS)
- KITTI dataset, 2D object detection subset — classes: Car, Pedestrian, Cyclist
- One detection model (YOLOv8n, fine-tuned), one distance-estimation module (geometric, not learned), one lane-detection module (classical CV)
- Full export chain: PyTorch → ONNX → TensorRT (FP32 and FP16)
- A real C++ inference pipeline (not just a Python demo) running on the user's local GTX 1650
- A rigorous cross-runtime benchmark: PyTorch (GPU) vs. ONNX Runtime vs. TensorRT-FP32 (C++) vs. TensorRT-FP16 (C++)
- Tests, CI, Docker, and a documented writeup with a demo video


### Stack

| Layer | Tools |
|---|---|
| Training / perception | Python, PyTorch, Ultralytics YOLOv8, OpenCV (Python), NumPy |
| Distance estimation | Classical geometry using KITTI calibration matrices (no additional training) |
| Lane detection | Classical CV (perspective warp, thresholding, Hough transform / polynomial fit) |
| Export | ONNX, `onnxruntime` (for validation) |
| Deployment | C++, CMake, TensorRT, CUDA, OpenCV (C++) |
| Testing | pytest (Python), a basic smoke test for the C++ binary |
| CI | GitHub Actions (lint + unit tests; GPU-dependent steps documented as out-of-CI-scope) |
| Containerization | Docker — one image for training/export, one CUDA/TensorRT image for deployment |
| Target hardware (this build) | NVIDIA GeForce GTX 1650, 4GB VRAM, Turing architecture (no Tensor Cores) |

### Suggested repo structure

```
EdgeLentils/
├── data/
│   ├── raw/                  # KITTI download lands here (gitignored)
│   └── processed/            # YOLO-format labels, train/val split
├── perception/
│   ├── convert_labels.py     # KITTI -> YOLO format
│   ├── train.py
│   ├── eval.py                # mAP evaluation
│   ├── distance.py            # geometric distance estimation + validation
│   └── lanes.py                # classical CV lane detection
├── export/
│   ├── to_onnx.py
│   ├── validate_onnx.py       # numerical parity check vs PyTorch
│   └── build_trt_engine.py    # FP32 / FP16 engine builder
├── deployment/                # C++ project
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp
│   │   ├── trt_engine.cpp/.h
│   │   ├── preprocess.cpp/.h
│   │   ├── postprocess.cpp/.h  # NMS, box decode
│   │   └── overlay.cpp/.h       # draw boxes, distance, lane overlay
│   └── tests/
├── benchmarks/
│   ├── run_benchmarks.py       # orchestrates all 4 runtime comparisons
│   └── results/                 # benchmark output tables/plots
├── tests/                       # pytest suite for Python components
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.deploy
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md          # diagram + explanation
│   └── results.md                # final benchmark tables, demo link
└── EdgeADAS_Project_Plan.md      # this file
```

---

## 2. Deliverables Checklist

### Phase 1 — Data pipeline
- [x] KITTI 2D object detection data downloaded (images + labels + calibration files)
- [x] KITTI label format converted to YOLO format
- [x] Classes filtered/mapped to Car, Pedestrian, Cyclist
- [x] Train/val split created — deviated from 80/20: primary split is the published Chen/3DOP 3712/3769 (sequence-disjoint, no temporal leakage across driving sequences); a class-stratified random 80/20 (5985/1496) is kept alongside it to quantify the leakage gap
- [x] Calibration parsing utility written and unit tested (reads KITTI's calibration matrices correctly)
- [ ] Repo skeleton created, first commit pushed — skeleton created (`perception/`, `tests/`, `docs/`); commit/push not yet done, pending your go-ahead

### Phase 2 — Perception model
- [ ] Training config defined (model size, image resolution, batch size — tuned to fit 4GB VRAM)
- [ ] YOLOv8n fine-tuned on the KITTI subset
- [ ] Baseline mAP@0.5 computed on val set and recorded
- [ ] Training run is reproducible from a single config file / command

### Phase 3 — Auxiliary modules
- [ ] Distance estimation implemented (bounding box bottom-center + camera calibration + flat-road assumption → real-world distance)
- [ ] Distance estimation accuracy validated against KITTI's ground-truth 3D annotations (report mean error, not just "it works")
- [ ] Lane detection pipeline implemented (perspective warp → threshold → fit)
- [ ] FCWS/LDWS warning logic implemented (simple rule-based thresholds, clearly documented)

### Phase 4 — Export & optimization
- [ ] Model exported to ONNX
- [ ] ONNX output numerically validated against PyTorch output (documented tolerance)
- [ ] TensorRT engine built at FP32
- [ ] TensorRT engine built at FP16
- [ ] mAP confirmed preserved (within acceptable tolerance) after each conversion step
- [ ] Exact CUDA / cuDNN / TensorRT version combination pinned and documented

### Phase 5 — C++ deployment
- [ ] CMake project set up, links TensorRT + CUDA + OpenCV successfully
- [ ] TensorRT engine loads and runs a single inference correctly (bare loop, before features)
- [ ] Preprocessing implemented in C++ (resize, normalize, format conversion)
- [ ] Postprocessing implemented in C++ (box decode, NMS)
- [ ] Distance + lane warning logic ported into the C++ pipeline
- [ ] Real-time video loop working end-to-end (video in → detections + warnings overlaid → display/output)

### Phase 6 — Benchmarking
- [ ] Benchmark harness built, running the same input video/images through all 4 runtimes
- [ ] Results collected: PyTorch (GPU), ONNX Runtime, TensorRT-FP32 (C++), TensorRT-FP16 (C++)
- [ ] Latency and FPS reported for each
- [ ] Accuracy parity confirmed across all 4 (detection outputs shouldn't meaningfully diverge)
- [ ] Results table/plot generated for the writeup

### Phase 7 — Testing & CI
- [ ] pytest suite covering: label conversion, distance math, NMS logic
- [ ] Basic smoke test for the C++ binary (runs without crashing on a known input)
- [ ] GitHub Actions workflow: lint + Python unit tests on every push
- [ ] README notes clearly that GPU/TensorRT steps require a GPU runner and are outside current CI scope

### Phase 8 — Docker
- [ ] `Dockerfile.train` — training/export environment, reproducible
- [ ] `Dockerfile.deploy` — CUDA/TensorRT deployment environment, reproducible
- [ ] Both images build successfully from a clean clone

### Phase 9 — Documentation & demo
- [ ] Architecture diagram (data flow: camera → detection → distance/lane → warnings → display)
- [ ] README with setup instructions, results table, and how to reproduce
- [ ] `docs/results.md` with final benchmark numbers and honest limitations section
- [ ] Demo video recorded (live detection + lane overlay + warnings, ideally slow-vs-fast side by side)
- [ ] "Future work" section written (see Section 3 below — this is where those ideas belong in the writeup)

---

## 3. Taking It Further (deliberately out of scope for now)

These are real next steps, not abandoned ideas — but they come *after* the core build ships, and only if there's time or interest left over. Do not pull these into the 8-week core scope.

- **Cloud GPU Tensor Core benchmarking** — rent a Tensor-Core GPU (e.g. via RunPod/Vast.ai/Lambda Labs, T4/L4-class) to add a genuine Tensor Core FP16/INT8 comparison row to the benchmark table. Cheap, low-risk, no purchase needed — the most likely near-term add-on.
- **Jetson deployment** — either buy a Jetson Orin Nano (note: price recently jumped to ~$399, plus Egypt shipping/customs risk) or rent one hourly via an emerging remote-hardware service (needs vetting for legitimacy/reliability before trusting it with real work). This is the step that would make the "edge deployment" claim fully literal rather than "designed for, validated on desktop." Real Tensor Cores here too, so INT8 quantization becomes worth doing.
- **INT8 quantization** — only pays off on hardware with Tensor Cores (cloud GPU or Jetson), so it's paired with one of the two options above, not the GTX 1650 build.
- **nuScenes / multi-camera / LiDAR fusion extension** — a genuinely harder, different problem (360° multi-sensor fusion, BEV representation) — the natural "v2" if this project leads somewhere, and closer to what valeo.ai's own published research looks like.
- **Learned lane detection** — replace the classical CV lane pipeline with a lightweight segmentation model, if the classical approach proves too brittle in testing.
- **ROS2 / BoilerHawk integration** — folding this perception stack into the existing BoilerHawk drone pipeline as a "BoilerHawk 2.0"-style extension, if there's appetite to revisit that project later.

---

## 4. Timeline

**Start:** Monday, August 3, 2026 · **Target completion:** Sunday, September 27, 2026 (8 weeks, ~20-25 hrs/week) · **Buffer:** a few days before senior year coursework realistically starts.

### Learning roadmap (topics genuinely new to the user — budget real time for these, don't rush)

- KITTI dataset format & the pinhole camera calibration model (intrinsics/extrinsics, projection math)
- Ultralytics YOLOv8 training workflow, config system, transfer learning/fine-tuning specifics
- Monocular distance estimation geometry (ground-plane assumption, projecting pixels to real-world distance)
- Classical lane-detection CV techniques (perspective/bird's-eye warp, color/gradient thresholding, Hough transform or sliding-window polynomial fit)
- ONNX export mechanics and common opset/numerical-parity pitfalls
- TensorRT concepts: engine building, precision modes (FP32/FP16), profiling, workspace memory
- CUDA fundamentals — enough to understand memory management in a real-time inference loop (not full CUDA kernel programming)
- Real-time video pipeline architecture (buffering, latency budgeting)

*(Docker, GitHub Actions, pytest, and C++/CMake basics are skipped here — already solid from prior projects.)*

### Week-by-week

**Week 1 (Aug 3–9) — Setup & data**
Learn: KITTI format, calibration math. Build: environment setup (PyTorch+CUDA matched to driver, Ultralytics, OpenCV), download KITTI, label conversion, train/val split, repo skeleton.
→ Completes Phase 1.

**Week 2 (Aug 10–16) — Baseline detector**
Learn: Ultralytics training workflow, fine-tuning mechanics. Build: train YOLOv8n on the KITTI subset, get baseline mAP.
→ Completes Phase 2.

**Week 3 (Aug 17–23) — Distance + lanes**
Learn: monocular geometry, classical lane-detection CV. Build: distance estimation module + validation, lane detection pipeline, warning logic.
→ Completes Phase 3.

**Week 4 (Aug 24–30) — Export & TensorRT**
Learn: ONNX export pitfalls, TensorRT concepts (precision modes, engine building). Build: ONNX export + validation, TensorRT FP32 and FP16 engines, pin all library versions.
→ Completes Phase 4.

**Week 5 (Aug 31–Sep 6) — C++ inference core**
Learn: TensorRT C++ API, enough CUDA to reason about memory/buffers. Build: CMake project, engine loading, bare inference loop working end-to-end.
→ Starts Phase 5.

**Week 6 (Sep 7–13) — Full pipeline + benchmarking**
Build: pre/postprocessing in C++, port distance/lane logic in, complete real-time loop; build benchmark harness across all 4 runtimes.
→ Completes Phase 5 and Phase 6.

**Week 7 (Sep 14–20) — Tests, CI, Docker**
Build: pytest suite, C++ smoke test, GitHub Actions workflow, both Dockerfiles.
→ Completes Phase 7 and Phase 8.

**Week 8 (Sep 21–27) — Documentation, demo, buffer**
Build: architecture diagram, README, results writeup with honest limitations, demo video recording, general polish. This week also absorbs any slippage from earlier weeks.
→ Completes Phase 9.

---

## 5. Hardware & Environment

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4GB VRAM, Turing architecture — **no Tensor Cores**, so FP16 gains come from kernel fusion/memory traffic reduction, not dedicated FP16 math units. Frame benchmark results accordingly (honest, not overstated). |
| CUDA / cuDNN / TensorRT versions | **TBD — pin during Week 1** based on the actual installed driver version. Do not assume specific versions; check compatibility tables before installing. |
| Model sizing constraint | 4GB VRAM → YOLOv8n (not larger variants), modest input resolution (e.g. 640×640 or lower), small batch size (~8–16) — confirm empirically in Week 2, adjust if OOM. |
| OS | Confirm and document in README (affects CUDA/TensorRT install steps). |

---

## 6. Definition of Done

The core build is complete when:
1. Every checkbox in Section 2 is checked.
2. The benchmark table shows real, honestly-reported latency/FPS/accuracy numbers across all 4 runtimes on the GTX 1650.
3. A fresh clone + Docker build + documented steps reproduces the pipeline without hidden manual fixes.
4. The demo video and README are something that could be shown in an interview without caveats beyond what's already written in the limitations section.
