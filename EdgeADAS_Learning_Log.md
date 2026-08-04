# EdgeADAS — Learning Log

A running list of concepts learned while building this project, as quick intuitive notes for later review — not detailed explanations, just enough to jog the memory and keep the vocabulary straight. Newest entries added at the bottom of each section as the project progresses.

---

## Project shape

- The project is two acts: perception (train something that can see) and deployment (make it fast enough to matter). The deployment half — TensorRT, C++, real benchmarks — is the actual differentiator, not the modeling.
- IoU (Intersection over Union): how much a predicted box overlaps the true box — the standard way to define "close enough."
- mAP (mean Average Precision): the detector's overall report card — how often it's both correctly placed and correctly labeled, averaged across classes.
- NMS (Non-Max Suppression): cleanup step that removes duplicate overlapping boxes for the same object.
- Fine-tuning: continuing to train an already-competent (pretrained) model on a specific task, instead of starting from zero.
- Epoch: one full pass through the entire training dataset.
- Inference: running an already-trained model on new data (as opposed to training it).

## Camera geometry & distance estimation

- Calibration files = the exact mathematical description of how a specific camera turns the 3D world into a 2D image. Intrinsics = lens properties (focal length, optical center); extrinsics = where the camera sits/points.
- Pinhole camera model: the simplified physics (light through a single point) that all this projection math is built on.
- fx / fy: focal length, tracked separately for horizontal/vertical since real lenses aren't perfectly symmetric.
- cx / cy: pixel coordinates of the lens's true optical center — the real "zero point" for measuring offsets, not necessarily the image's literal center.
- Flat-ground assumption: since the camera's height H above the road is known, any point where an object touches the ground has the same vertical offset from the camera (Y = H), no matter how far away it is.
- Distance formula: **D = fy × H / (v − cy)**, where v is the pixel row of the object's ground-contact point (box bottom). Bigger gap from cy (further down-frame) = closer object; v approaching cy (near the horizon) = distance approaching infinity — makes physical sense.
- This only holds for a flat road and a level camera — real-world error from that assumption is exactly what validating against KITTI's ground-truth 3D positions is meant to catch.

## KITTI dataset specifics

- KITTI's `training` folder = labeled data we split ourselves. `testing` folder = unlabeled, reserved for the official leaderboard — unusable for our own evaluation.
- Only needed: left color images + calibration matrices + training labels. Skipped: right color (stereo pair, not our approach), Velodyne (LIDAR point clouds — deliberately out of scope), grayscale (different benchmark task).
- KITTI label format: text class name + absolute pixel box corners (left/top/right/bottom) + 3D fields. Different from YOLO format: integer class index + normalized (0–1) center-x/center-y/width/height.
- Sequence leakage: KITTI images come from continuous driving sequences, so a naive random train/val split puts near-duplicate consecutive frames on both sides, inflating validation mAP. The Chen/3DOP split avoids this by keeping whole sequences on one side, and is the field's de facto standard — makes our mAP comparable to published numbers.
- Class merging (Van/Truck → Car, Person_sitting → Pedestrian, Cyclist stays, Misc/Tram/DontCare dropped): not just "easier" — gives each class enough examples to learn reliably, matches what actually matters for collision warning, and keeps our scheme comparable to the field standard. DontCare isn't a real class — it's a "don't penalize detections here" masking instruction, not an object.
- KITTI images are ~1242×375 (3.3:1 — much wider than tall). Forcing that into YOLO's square 640×640 input via letterboxing wastes a large share of the canvas on padding and unnecessarily shrinks real content. Flagged as a Phase 2 optimization, not fixed now — protect a working baseline first.

## Tooling & engineering

- Junction/symlink: a folder that looks real to any program reading it, but doesn't duplicate the underlying files — like a desktop shortcut. Used to satisfy Ultralytics' hardcoded expectation that "images"/"labels" appear literally in the folder path, without physically copying ~5.9GB of data.
- Ultralytics infers a label file's path by swapping the literal word "images" for "labels" in the image's path — a fixed convention, not something you configure per-dataset.
- ONNX: a universal file format for trained models — like a PDF for AI models, readable by many different tools regardless of what trained them.
- TensorRT: NVIDIA's tool that recompiles a model specifically for the exact GPU it'll run on, trimming/fusing operations for speed.
- FP32 vs FP16 (precision): fewer bits per number = faster, less memory, usually negligible accuracy loss for neural nets — but real FP16 speed gains need Tensor Cores, which the GTX 1650 (Turing) doesn't have.

## Claude Code workflow

- `CLAUDE.md` auto-loads at the start of every Claude Code session in a project folder — keep it short and use `@file` imports to pull in longer specs (like the project plan), since shorter instruction files get followed more reliably.
- Plan mode: Claude describes its approach before touching any files, shown as a commentable document — good for phase-by-phase check-ins on a tightly scoped project.
- `opusplan`: uses Opus for reasoning/planning, drops to Sonnet for the actual code-writing — a good fit for a project mixing genuinely new territory (TensorRT, CUDA, camera geometry) with a lot of routine implementation.

## Hardware & deployment decisions

- Cloud GPU rental (RunPod/Vast.ai/Lambda-type services) vs buying a Jetson: cloud gives a cheap Tensor-Core benchmark data point; a Jetson makes the "edge deployment" claim literal. Both deliberately deferred past the core 8-week build to protect the timeline.
