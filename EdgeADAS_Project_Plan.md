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
- [x] Repo skeleton created, first commit pushed — skeleton created (`perception/`, `tests/`, `docs/`);

### Phase 2 — Perception model
- [x] Training config defined (model size, image resolution, batch size — tuned to fit 4GB VRAM) — `configs/train_baseline.yaml`; batch=16/workers=3 pinned by a real VRAM+RAM probe, not guessed (see file comments)
- [x] YOLOv8n fine-tuned on the KITTI subset — 67 epochs (early-stopped, patience=30, best at epoch 37), Chen split only; random-split leakage-comparison run still deferred
- [x] Baseline mAP@0.5 computed on val set and recorded — square-640 baseline: overall mAP50=0.569, mAP50-95=0.340 (Car 0.875 / Ped 0.518 / Cyc 0.315). Diagnosed as class imbalance (20x fewer Cyclist boxes than Car) + 640-square downscaling hurting small/thin objects, not a bug. Full numbers in `metrics_baseline_square640.json`.
  - **UPDATE — primary model promoted:** the rect-1024 fine-tune (`runs/detect/rect1024_finetune/weights/best.pt`, fine-tuned from the square-640 baseline checkpoint rather than retrained from scratch) is now the primary model. It improved every class — Car 0.875→**0.902** (+0.027), Pedestrian 0.518→**0.564** (+0.046), Cyclist 0.315→**0.421** (+0.106, the largest gain, as predicted for the class hurt most by downscaling) — overall mAP50 0.569→**0.629**, mAP50-95 0.340→**0.387**. Real, meaningful improvement, though visibly still room to close the gap further (per direct review of prediction images) — class-imbalance mitigation (Section 3) is the queued next lever if that's pursued. Full numbers now in `metrics.json`; square-640 baseline kept for reference in `metrics_baseline_square640.json`.
- [x] Training run is reproducible from a single config file / command — `python -m perception.train --config configs/train_baseline.yaml`

### Phase 3 — Auxiliary modules
- [x] Distance estimation implemented (bounding box bottom-center + camera calibration + flat-road assumption → real-world distance) — `perception/distance.py`, D = fy×H/(v−cy). Camera height H=1.65m is a researched literature constant (Chen et al.'s Mono3D, same lineage as the Chen split), not a per-frame calib value — KITTI's calib files don't encode camera-to-road height.
- [x] Distance estimation accuracy validated against KITTI's ground-truth 3D annotations (report mean error, not just "it works") — on the Chen val split (n=19,855, ground-truth boxes not detector predictions, so this isolates geometry error): overall median error 2.99m, p95 24.54m (mean 12.58m — tail-dominated by ~36 objects whose box bottom lands within 2px of the horizon row, a genuine numerical singularity of the flat-ground formula at long range, not a bug; flagged explicitly in the report rather than hidden). Per-class median: Car 3.21m, Pedestrian 1.66m, Cyclist 3.00m. Robust (median) implied camera height across all samples: 1.789m vs the shipped 1.65m constant — close enough to confirm the method and constant are fundamentally sound.
  - **UPDATE — finer-grained bucket breakdown**, replacing the original near/mid/far(0-20/20-40/40+) split: `<10m` n=3338 median=1.19m (relative error 38%), **`10-25m` n=7428 median=2.07m (relative error 22%)**, `25-50m` n=6846 median=6.06m (50%), `50m+` n=2243 median=15.23m (70%). The `10-25m` bucket — the one that straddles the 15m FCWS threshold — is the best-performing range in both absolute and relative terms, closest to the ~0.1m/m figure reported for comparable geometric methods in the literature. The `<10m` bucket's elevated relative error is counterintuitive (pixel-noise error should only grow with distance); investigated and found 49% of `<10m` objects carry meaningful KITTI truncation (image-edge clipping) vs. 4% at 10-25m, correlated with error at r=0.42 — a real, partial explanation, not the whole story (truncation-free close objects still show ~1.1m median error). Full numbers in `distance_validation.json`.
  - **UPDATE — camera roll correction added**: user noticed distance was underestimated for objects with large lateral offset and asked whether a Pythagorean radial-distance term was missing. Checked: it wasn't — both our estimate and KITTI's ground truth are forward depth (Z), not radial distance, so that comparison would have compared two different quantities. The *real* effect found on investigation: error correlates with signed lateral position (r≈0.27 on a clean, non-truncated, non-near-horizon subset), antisymmetric — one side underestimates, the other overestimates by a similar amount — the signature of camera roll (the camera's image-row axis isn't quite aligned with true vertical). Fit `CAMERA_ROLL_RAD ≈ -1.4°` by minimizing val-set error; `estimate_distance_for_box`/`estimate_position_for_box` now apply `D = H / [(v−cy)/fy − (u_center−cx)/fx·roll]`, reducing exactly to the plain formula at roll=0. Cuts error in the far-lateral tails (`|X|>10m`) by ~30% (e.g. `[10,30)m` bucket: median error 9.9m→6.7m); negligible near X=0 where most objects sit, so overall median only moves 2.99m→2.90m. Unlike `CAMERA_HEIGHT_M`, this angle is a calibration fit against this project's own val set, not an independently-cited literature constant — documented as such in the module docstring rather than presented as equally well-sourced. Robust implied H also moved closer to the shipped constant (1.789m→1.737m vs 1.65m), corroborating the fit.
- [x] Lane detection pipeline implemented (perspective warp → threshold → fit) — `perception/lanes.py`: trapezoid ROI warp → combined HLS S-channel + Sobel-x gradient threshold → sliding-window 2nd-order polynomial fit. No ground truth exists for lanes in this KITTI subset (lane labels are a separate KITTI benchmark, out of scope), so validated visually: 4 of 5 spot-checked frames tracked the lane convincingly, including through curves; 1/5 (a glare-heavy intersection with no real markings) failed — an honest, expected limitation of classical thresholding under difficult lighting, not a crash. Matches the pre-existing "Learned lane detection" Section 3 bullet flagging this exact brittleness risk.
  - **UPDATE — sanity-check gating added, fixing near-constant LDWS false triggers**: user reported LDWS was "completely not working." Root cause confirmed empirically: `fit_lane_polynomial` had no quality gate — across a 400-frame sample it returned *some* fit for both boundaries 100% of the time, even when the underlying pixels were curb shadow or a parked car's edge rather than a real lane marking (many KITTI residential streets have no painted lines at all). Researched how other classical implementations handle this (the "Advanced Lane Finding" reference lineage this pipeline descends from): reject fits post-hoc if the two boundaries aren't roughly parallel or imply an implausible width, rather than accepting anything `np.polyfit` can produce. Implemented as `_lanes_are_plausible()` (checks width consistency at two rows, rejects crossed/too-narrow/too-wide/non-parallel fits) plus a raised minimum-pixel bar (`MIN_LANE_PIXELS=50`, up from an effective 3) in `fit_lane_polynomial`; on rejection, `detect_lanes()` now returns `None` for both fits, which downstream code already handled correctly. Re-measured on the same 400 frames: 100%→86.5% "found", with the 13.5% now correctly rejected — spot-checked several rejected frames against their un-gated raw fits and confirmed they were genuinely bad (one collapsed to a triangle pointing at a parked car on a street with no lane markings at all); spot-checked accepted frames and confirmed they track real markings well. Considered the heavier fix (replacing the classical pipeline with a lightweight learned model, e.g. Ultra-Fast-Lane-Detection) — real option, matches the pre-existing "Learned lane detection" Section 3 bullet, but needs labeled lane training data this project doesn't have; deferred since the cheaper fix resolved the diagnosed root cause.
  - **UPDATE — lane search redesigned from argmax to scored candidates**: user diagnosed that picking start columns by raw pixel count structurally favors continuous features (fences, guardrails, curbs) over dashed paint — confirmed empirically (on one highway frame the right-half histogram's six highest columns were all the guardrail at 0.99w). Redesign per user spec: up to 4 local histogram peaks per side (min-separation-enforced so one dominant edge can't consume all slots), each walked/fitted/scored per-line, then all left×right pairs scored and the best pair wins. Locked signals & weights (tuned on a 12-frame set of known-failure + known-good frames, not guessed): per-line = 0.4·curve-coherence + 0.4·center-proximity(soft Gaussian at 0.32w/0.68w, σ=0.15w) + 0.2·pixel-support; pairwise = 0.4·line-avg + 0.3·width-plausibility(plateau 0.20w–0.45w) + 0.3·parallelism(width-CV). Honest signal findings from tuning, per the pre-registered flag: **(1) dash-likelihood as originally specified was unmeasurable and actually inverted in practice** — road-texture threshold noise fills dash gaps when measured over the walk corridor (real dashed lines read ~98% covered, off-road clutter ~83%), and warped dashes read dense even in a tight band; replaced with curve-coherence (pixels lying consistently along the fitted curve: real boundaries measured 0.63–0.99 vs scattered clutter 0.19–0.53), which is what the data actually supports. **(2) No absolute accept/reject evidence exists in these geometric signals**: winning-pair scores on good frames (0.80–0.95) and paint-free-street garbage (0.80–0.96) overlap completely — measured, not assumed — and pixel-brightness also fails to separate (curbs are dark-but-real, parked cars are bright-but-clutter). Consequence: the 86.5%/13.5% rejection rate from the earlier gate is gone (100% found again) — that rejection was partly accidental (the argmax search produced geometrically broken fits on paint-free streets which the gate caught; the scored search optimizes for exactly the geometry the gate checks). Known remaining limits: paint-free streets now return a plausible-geometry "drivable corridor" reading rather than a rejection, and genuinely faint dashes can still lose to crisp clutter edges (upstream threshold/warp signal-quality ceiling, out of scope for this redesign). Rejection-restoration options (vehicle-overlap veto using the detector; learned lane detection) identified and pending user direction after visual review of the regenerated demo set. Window margin also made resolution-relative (`WINDOW_MARGIN_FRACTION=0.064` ≡ the old 80px on 1242px frames) after the fence-vs-dash unit test exposed corridor merging of nearby features on smaller inputs.
- [x] FCWS/LDWS warning logic implemented (simple rule-based thresholds, clearly documented) — `perception/warnings.py`: FCWS triggers under a single 15m distance threshold (not per-class/speed-based — explicitly out of scope), LDWS triggers past a 0.3m lane-center-offset threshold. LDWS is a per-frame instantaneous check, not temporal drift tracking — the KITTI 2D detection frames used are independently-sampled stills, not continuous video, so there's no trend to track (matches the pre-existing "Upgrade LDWS to trend based" Section 3 bullet, which already correctly identifies this as future work contingent on different data).
  - **UPDATE — FCWS made lane-aware via a fixed ego-path corridor**: distance alone can't distinguish an in-lane threat from a same-depth object in an adjacent lane (the depth geometry is correct — v−cy is genuinely shared — the gap was using depth alone as the *relevance* filter). Added lateral position `X = (u_center−cx)·Z/fx` (`perception/distance.py:estimate_position_for_box`, same pinhole family as the depth formula) and a fixed corridor check, `abs(X) < CORRIDOR_HALF_WIDTH_M`, independent of `lanes.py` succeeding on a given frame. `CORRIDOR_HALF_WIDTH_M` = half of KITTI's actual recording vehicle width (VW Passat B6, 1.82m — Wikipedia, cross-checked auto-data.net) + a 0.5m margin. The half-vehicle-width term is cited (matches a real FCW patent's stated baseline rule); the margin is an honest, reasoned engineering judgment, not an independent citation — flagged as such rather than dressed up as research. Documented limitation: the corridor assumes the vehicle is travelling straight, not that the road is straight — no steering/yaw/IMU data exists in this dataset to curve it to real heading; visually confirmed honest in the demo renders below (corridor diverges visibly from the actual lane on a curved-road frame). Validated visually via `perception/demo.py` across 13 real frames using the promoted detector: close-but-laterally-offset objects (parked cars, roadside pedestrians) correctly excluded even at <10m; genuine in-path threats (a following car, a pedestrian crossing ahead) correctly trigger; FCWS confirmed working on a frame where lane detection itself failed (glare/intersection), demonstrating the two are now decoupled.

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
- **1024x32+ no mosaic training** — Try a higher a resolution for the finetuning pipeline, risk losing the benefits of mosaic augmentation. *(Done, promoted to primary model as of Phase 2 — fine-tuned the existing baseline checkpoint at rect 1024 instead of retraining from scratch, so mosaic's augmentation benefit from the original training run wasn't lost. Real gains across all classes, Cyclist most (+0.106 AP50) — see Phase 2 checklist. Improvement is real but partial; still room to close the gap further.)*
- **Class-imbalance mitigation** — Pedestrian/Cyclist AP50 (now 0.564 / 0.421 post rect-1024, up from 0.518 / 0.315) still lag Car (0.902) largely because Car has 20x more training boxes than Cyclist. The rect-1024 fine-tune helped but didn't close the gap — worth trying next: class-weighted loss, oversampling rare-class frames, or targeted copy-paste augmentation for Pedestrian/Cyclist. Not implemented now.
- **Build our own implementation of mosaic outside Ultralytics** — this way we can implement mosaic and get the benefits it offers without having to submit to the
limitations of Ultralytics when it comes to rectangular images. The whole point of turning off mosaic in rectangle mode is that is keeps the aspect ration ordering
in place. Since all Kitti images are the same size and aspect ration, we don't need to keep this sorting anyways. The workaround is that we build our own mosaic script and feed it to the Ultrlytics pipeline as normal rectangular images.
- **Assess the trade-offs if we upgrade to a non-nano YOLO model** — Since we very much care about performance and keeping the model lightweight, upgrading to a model with more params will need some research first on how slower it will be in inference.
- **Upgrade the LDWS to trend based instead of static check** — The version of Kitti we have was meant for object detection, this means the images are not sequential, they are independent samples from different scenes. LDWS is usually a temporal thing, it detects movement overtime. We currently have it as a static check.
- **Curve the FCWS ego-path corridor to real vehicle heading** — the corridor added in Phase 3 assumes the vehicle is always travelling straight, which is honest given this dataset but wrong during an actual lane change or turn. Fixing this needs steering angle, yaw rate, or IMU data to bend the corridor to true heading — requires a continuous-driving KITTI product (e.g. the raw/tracking data, not the 2D object detection subset this project uses), not just more code. Not attempted now.

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
