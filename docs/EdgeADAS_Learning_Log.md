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

## Resolution fix (Phase 2) — corrected after verifying installed Ultralytics

- Correction to the earlier plan: a fixed non-square `imgsz` is actually rejected outright for training in the installed Ultralytics version — the only way to get a non-square shape is `rect=True`, which force-disables mosaic/mixup/cutmix augmentation. Good example of why checking the actual installed library beats trusting general docs/memory.
- This square-only training constraint is a long-standing general Ultralytics design decision (true since early YOLOv8 releases, still true now) — not a version bug. It's not actually required by the network's stride math (which works fine for any properly-divisible rectangle); it comes specifically from mosaic's default 2×2 tiling being implemented as a square canvas, which is why `rect=True` (a non-square shape) and mosaic can't coexist. The deeper reason: rect training pre-sorts the whole dataset by aspect ratio, groups similar shapes into fixed, non-shuffled batches, and gives each batch its own custom padded shape — while mosaic needs to freely grab 3 random images from anywhere in the dataset at every step and glue them onto one universally fixed square canvas. Order-matters-and-shape-varies-per-batch vs. order-doesn't-matter-and-shape-is-always-the-same are structurally incompatible, not just a formatting mismatch — so Ultralytics just hard-disables mosaic under rect rather than reconciling them. Rect was originally built for efficient validation/inference batching, not augmented training, which is why the two were never designed to coexist.
- Nuance: on a dataset like KITTI where every image already shares ~the same aspect ratio, rect's sort-and-group-by-aspect-ratio step is genuinely a no-op (nothing to usefully group) — that machinery is really designed for datasets with mixed shapes like COCO. But the actual conflict with mosaic isn't the sorting itself, it's an unconditional downstream side effect: `rect=True` disables shuffling and locks batch order for the whole run regardless of whether the sort did anything useful, and that fixed-order requirement is what collides with mosaic's need for free random sampling every step. There's also no simpler workaround available — a plain non-square `imgsz` is rejected outright outside of rect mode too, so rect's side effects come bundled in even when its sorting logic buys nothing.
- The divisible-by-32 rule itself is still architecturally fundamental (stride-based multi-scale feature fusion), independent of the square-vs-rect question.
- Mosaic: stitches 4 images into a collage-style training image (labels repositioned accordingly) — creates huge combinatorial variety from a fixed dataset and builds robustness to partial/occluded/oddly-scaled objects. One of YOLO's biggest single mAP contributors, especially valuable on a modest dataset.
- MixUp: pixel-blends two whole images together (like a double exposure) with blended labels — gentler regularizer, discourages brittle decision boundaries.
- CutMix: pastes a cut rectangular patch from one image directly onto another (unblended) — forces recognition from an object's own features rather than its usual surrounding context.
- Decision: Square 640 + mosaic for the first real training run (standard, best-documented path, directly comparable to published YOLOv8n benchmark numbers, lowest execution risk — some practitioners also report `rect=True` training being genuinely less stable in practice, not just missing augmentation). Rect 1024×320 (no mosaic) is mathematically more efficient — fewer total pixels than square-640 *and* better-preserved object scale — but loses proven augmentation on a modest dataset; logged as a deliberate follow-up experiment once the full pipeline (train → export → C++ → benchmark) is proven working end to end.
- Square 960 + mosaic ruled out: batch size collapses to ~4 on 4GB VRAM (destabilizes BatchNorm statistics), and 2.25× inference cost directly hurts the project's core FPS differentiator — without even solving the aspect-ratio mismatch better than the rect option.
- Whatever resolution is finalized must stay fixed through to the TensorRT engine in Phase 4-5.

## System resources & batch sizing

- GPU VRAM is a hard, actively-enforced limit — exceeding it throws an immediate, clean CUDA out-of-memory error. System RAM has no such hard wall — when it runs low, the OS falls back to paging (writing memory to disk and reading it back), which doesn't crash the program, just makes it crawl silently. This is why a too-large batch under tight system RAM "fails slowly" instead of erroring out fast.
- Dataloader workers: separate processes that load/preprocess the *next* batch in parallel while the GPU trains on the current one, so the GPU never sits idle waiting for data. Each worker needs its own memory (decoded images, augmentation buffers), and larger batches mean more memory needed per worker — this is the direct link between low system RAM and batch size limits.
- When a resource constraint's practical limit depends on real runtime conditions (not just theory), a short bounded probe (a couple minutes, a few candidate values, stop before any real training) beats both guessing conservative (leaves performance on the table) and trusting an automatic heuristic blindly (can overshoot, and an overshoot is worse exactly when the failure mode is "slow thrash" rather than "clean crash").

## Training config terms

- Epoch vs batch: a batch is one chunk of images processed per training step; an epoch is one full pass through the *entire* training set (made of many batches). `epochs: 100` means seeing all training images 100 times over, not 100 training steps.
- `cache`: whether to hold the dataset in memory/disk between epochs instead of re-reading from source files each time — faster, but costs real memory. Set false here specifically because of the tight system RAM situation.
- `amp` (Automatic Mixed Precision): runs most training ops in FP16, keeps numerically sensitive ones in FP32, automatically. On a Tensor-Core-less GPU (GTX 1650) the main benefit is memory savings, not a big speed boost.
- `seed`: fixes the starting point of all of training's "randomness" (shuffling, augmentation choices) so reruns follow the identical sequence — needed for reproducibility and for fair comparisons (e.g. Chen vs random split).
- `deterministic`: a separate concern from seed — GPU parallel execution can introduce tiny run-to-run numerical differences from thread-scheduling variability and floating-point non-associativity, even with a fixed seed. This flag forces only bit-for-bit-reproducible algorithm implementations, at a small speed cost.
- `close_mosaic`: disables mosaic for the final N epochs of a run, so the model spends its last stretch calibrating on plain, realistic (non-collaged) images instead of ending training still adapted to mosaic's collage look.
- `patience`: early stopping tied to validation performance — stops training if validation mAP goes N consecutive epochs without improving, rather than being a generic "stuck" timeout. Worth checking a default value actually fits within your total epoch count — a patience equal to or greater than total epochs never fires and silently does nothing.
- `device`: which GPU to use, by index (0 = first/only GPU) — stated explicitly to rule out any silent fallback to (much slower) CPU training.

## Phase 2 baseline results & sequencing decision

- Baseline (square 640 + mosaic, Chen split, early-stopped at epoch 67, peak at 37): Car AP50 0.875 (as predicted), Pedestrian 0.518, Cyclist 0.315 — overall mAP dragged down by class imbalance (Cyclist has ~20× fewer training examples than Car) compounded by small objects suffering most from the 640×640 mosaic shrinkage (~52% linear scale). A converged, working model, not a failed run — patience correctly caught a genuine plateau.
- Key realization: the rect 1024×320 (no-mosaic) comparison experiment and Phase 3 don't actually compete for resources — distance estimation (pure geometry) and lane detection (classical CV) need no GPU and no trained model to build/validate, so the comparison retrain can run in the background while Phase 3 development proceeds in parallel, rather than treating "defer improvement" vs "fix it first" as a real either/or.
- Good practice for distance validation specifically: test the geometric formula against KITTI's *ground-truth* boxes, not the detector's own output — isolates whether the geometry itself is correct, independent of detector recall, and sidesteps weak Cyclist/Pedestrian recall entirely for that validation step.
- Deliberately not reaching for other class-imbalance techniques (weighted loss, oversampling, focal loss) yet — one targeted experiment at a time, see what the resolution fix alone buys before adding more levers.

## Continuing training from a checkpoint (fine-tune phase)

- No need to retrain an identical square+mosaic baseline from scratch before the rect fine-tune phase — with `seed`/`deterministic` already set, a fresh retrain of the same config would reproduce essentially the same checkpoint anyway, making it redundant compute. Reuse the existing `best.pt` directly.
- Use `best.pt` (peak validation epoch), not `last.pt` (final epoch reached after plateauing) — you want to continue from the best-generalizing state, not the point training had already stopped improving from.
- The real thing to get right when continuing training on already-converged weights: use a deliberately reduced learning rate (and shorter epoch budget) for the continuation phase, not the same full-strength schedule as a fresh run — otherwise the optimizer can destabilize/forget what's already been learned before settling into a new minimum. This matters regardless of which checkpoint you start from.

## Working around the rect/mosaic conflict

- Ultralytics' *live* rect+mosaic conflict is real and confirmed by an actual bug report (someone set both `rect=True` and `mosaic=1.0`, expecting both to apply, and got poor results — Ultralytics silently disables mosaic under rect regardless of the flag, without making that obvious, so they likely trained with unintended zero augmentation).
- Real workaround: mosaic isn't Ultralytics-proprietary — it's a straightforward tile-N-images-and-adjust-labels technique, and standalone offline implementations already exist. Generate mosaic-combined images *once, offline*, as real files targeting our actual rectangular canvas shape (not Ultralytics' live per-step transform), add them to the training set, then train normally — sidesteps the internal conflict entirely since we're not asking the live pipeline to do both at once.
- Honest tradeoff: offline-generated mosaics are a fixed set of combinations (decided upfront) vs. live mosaic's fresh random draw every epoch — less diversity than the built-in version, but still real, meaningful augmentation.
- Bonus: writing it ourselves means we're not stuck with Ultralytics' square-oriented 2×2 grid — a tiling arrangement suited to KITTI's actual wide aspect ratio (e.g. 2 images side by side) could work better than the default anyway.

## FCWS / LDWS / LKAS and Phase 3 lane scoping

- FCWS (Forward Collision Warning): warns if closing the gap to the vehicle ahead too fast. LDWS (Lane Departure Warning): warns if drifting across a lane line without signaling. LKAS (Lane Keeping Assist): goes further than LDWS by actively steering back — not something this project implements, just the warning version.
- No ground truth exists for lane detection in the KITTI object-detection subset we downloaded (it labels cars/pedestrians/cyclists, not lane markings). KITTI's separate Road benchmark has real ground-truth masks but was deliberately deferred earlier to protect scope. Decision: validate via visual spot-checking across a diverse sample (legitimate for a deterministic, inspectable classical-CV pipeline) plus an optional small (~30-50 image) self-hand-labeled sample if a real quantitative number is wanted for the writeup — without taking on the full Road benchmark as new scope.
- LDWS is properly a temporal-trend detector (drift over consecutive frames), not a single-frame check. Correction to the earlier reasoning: it's not actually the Chen-split decision that blocks this — the base `image_2` file numbering (000000.png, 000001.png, ...) was never chronological order in the first place, even before any split. Evidence: KITTI released "3 temporally preceding frames" as a separate optional download specifically because the base image set doesn't include that context, and the object-detection benchmark's images are a sparse sample largely distinct from KITTI's actually-continuous tracking-benchmark video. Genuine sequential frames would require new scope (the preceding-frames download, or KITTI's separate raw-data category) — same "real extension, not core scope" pattern as the Road benchmark and nuScenes. Decision unchanged: instantaneous "how far off-center right now" check, explicitly documented as a simplification — just for a more precise underlying reason.

## Phase 3 results review

- Ground truth for distance validation: KITTI's LIDAR-measured 3D positions, independent of our formula — validating ground-truth boxes through our formula against LIDAR-measured positions is a genuinely independent check, not circular.
- 2.99m median distance error is in line with published results for this class of method (flat-ground/IPM-style geometric estimation) — one comparable study reports roughly 0.1m error per meter of true distance, and accuracy degrading with distance is a well-documented, inherent property of the geometry, not a flaw specific to our implementation. Better framing than a single aggregate number: report error broken down by true-distance buckets, since near-range accuracy (what FCWS's threshold actually depends on) matters far more than the far-range noise a blended median can hide.
- Real gap identified: the distance formula correctly ignores horizontal (lateral) position when computing depth — on a flat road, depth only depends on vertical pixel position, so two objects at the same depth in different lanes genuinely do share the same value, that's correct geometry, not a bug. But FCWS's decision logic (raw "is any detected object under 15m") doesn't currently account for lane membership, so it can't distinguish an in-lane threat from an adjacent-lane car at the same depth. Fix: compute lateral position X too (same pinhole math, using the box's horizontal center — cheap addition), and use `lanes.py`'s output to filter FCWS to lane-relevant objects — the ingredients for genuine BEV-style reasoning.

## Ego path corridor (fixed geometric fallback for FCWS)

- Since the camera is rigidly mounted to the car, "my own lane" (a fixed width centered on the vehicle) projects to a constant position relative to the camera regardless of whether painted lane markings exist — same idea as an "ego path corridor" used in real production ADAS, specifically valued for being robust to missing/faded markings.
- Cleanest implementation: not drawn pixel-space lines, but a check in real-world meters — reuse the lateral position (X) already being added for lane-aware FCWS, and compare |X| against half an assumed corridor width. Avoids extra geometric line-intersection code entirely.
- Sharper framing (important correction): FCWS is fundamentally about the car's actual physical path, not about "lanes" as a painted-road concept — "lane" only happens to be a good stand-in when it matches the car's real heading. Since the system has no steering/yaw/IMU data, "where the car is heading" collapses to "straight ahead of the fixed camera" purely from lack of sensing — same math as before, but call it an **ego path corridor**, not a lane corridor, to keep it conceptually distinct from `lanes.py`'s real lane-marking concept (which stays exactly as-is for LDWS). Honest limitation, correctly stated: assumes the car is currently going straight (violated by a real lane change or turn), not "assumes the road is straight."
- This mostly resolves Claude Code's FCWS fail-open/fail-closed dilemma: if FCWS's relevance check uses this fixed geometric corridor instead of live lane detection, it no longer depends on `lanes.py` succeeding that frame at all. The fail-open/fail-closed question becomes specifically an LDWS concern (which genuinely needs the real detected lines, since departure is about actual paint), not an FCWS one.

## Camera roll correction — results

- Clean before/after (same code, same val split, only `roll_rad` toggled) confirmed real improvement: mean error -9.9%, median -3.0%, p90 -8.4%, p95 -4.0% overall. Verified the underlying arithmetic independently — all correct.
- One honest mixed signal worth understanding, not just noting: in the FCWS-critical 10-25m bucket, mean error dropped 31% but median ticked *up* slightly (2.07m→2.21m). Likely explanation: the roll angle was almost certainly fit by minimizing a mean/squared-error-style loss, which is dominated by outliers — it optimizes hard for crushing the biggest errors and is nearly indifferent to small, already-fine typical-case errors. That fitting-objective choice predicts exactly this pattern (big wins on outlier-sensitive stats, flat-to-slightly-worse median), rather than it being an unexplained anomaly.
- Worth stating explicitly for a safety-relevant system: trading a few cm of typical-case accuracy for a meaningfully lower rate of bad outlier errors is arguably the right tradeoff for a collision-warning system specifically — the failure mode that matters is the rare badly-wrong estimate causing a missed warning, not small variation in an already-comfortable typical case.
- Still open: whether the roll correction changed any object's actual ego-path-corridor membership (in-path vs. filtered-out), not just its reported distance — a different, more operationally relevant question than the distance-accuracy numbers alone answer.

## Classical lane detection: pipeline internals & failure modes

- Perspective warp uses a fixed trapezoid guessed as fractions of image width/height — not derived from calibration, road geometry, or the vanishing point. Assumes a flat, straight, centered road on every frame regardless of intersections, curves, or hills.
- Thresholding combines HLS saturation (catches painted markings) OR Sobel-x gradient (catches edges) with fixed absolute cutoffs, no per-frame adaptation to lighting/shadows — anything satisfying either condition passes, including curbs, cast shadows, and object silhouette edges.
- Lane search: a histogram (column-wise pixel sum, bottom half of the warped mask) picks a starting x per side, then a 9-band sliding window tracks upward, recentering only if a band captures enough pixels; otherwise the window freezes in place rather than signaling lost track. All collected pixels across all bands get fit with a single 2nd-order polynomial.
- Known fragility: the starting-column histogram has no notion of "looks like a lane marking" vs. "brightest/most-present thing in frame" — a continuous edge (fence, railing, curb) easily outscores real paint. The existing sanity-check gate catches implausible final fits after the fact, but doesn't fix the upstream search itself.

## Reference project: Hough-transform lane detection (no neural network)

- Found a college-project writeup using Canny edges + probabilistic Hough transform + rho/theta clustering + angle-based left/right split, with heavy temporal stabilization (previous-frame fallback, weighted averaging, lane-switch detection via frame-to-frame lane-area changes).
- Critical mismatch: most of what makes that project's algorithm robust is temporal continuity across 30fps video — none of it transfers, since the Chen split is independently sampled, non-consecutive frames (the same property that avoided train/val leakage earlier). No "previous frame" exists to fall back on.
- What does transfer (per-frame, no continuity needed): bilateral filtering before edge detection (denoises while preserving edges, unlike Gaussian blur); an absolute angle-plausibility filter on candidate lines (complements the parallelism signal, which only compares left vs. right to each other, not either one against a plausible range on its own).
- Genuinely interesting idea, logged as a future escalation path rather than adopted now: Hough transform's voting mechanism naturally aggregates fragmented/dashed evidence onto a shared line hypothesis (multiple collinear dash segments vote for the same rho/theta bin), which is structurally well-suited to the dash-detection problem in a way raw pixel-count histograms aren't. Doesn't solve paint-vs-structure on its own, though — that project uses no color signal at all, purely grayscale edges plus geometry.
- Not adopted: their HAAR-cascade car detector (superseded by YOLOv8n already in the pipeline).

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

## Real-world sensor context

- Full self-driving fleets (Waymo-style) mostly sidestep monocular distance inference — LiDAR gets distance by timing a laser's bounce-back, radar gets it (plus closing speed) via Doppler shift. Neither is inferred from a 2D image.
- Vision-only systems (Tesla-style) still solve our exact problem, but with a learned monocular depth model — a network trained end-to-end to predict distance from the image, usually supervised by stereo/LiDAR data during training — rather than an explicit geometric formula.
- Mainstream consumer ADAS (the closest real analog to this project) commonly pairs one forward camera with radar: camera for classification, radar for a precise direct range/closing-speed reading — specifically because a pure geometric ground-plane estimate is fragile on hills, bumps, or any camera pitch change.
- Our geometric approach is real and legitimate, just the simplest/most interpretable option — right-sized for a portfolio project. Good interview line: the natural next step in production would be pairing it with radar or swapping in a learned depth model to fix the flat-road fragility.

## Hardware & deployment decisions

- Cloud GPU rental (RunPod/Vast.ai/Lambda-type services) vs buying a Jetson: cloud gives a cheap Tensor-Core benchmark data point; a Jetson makes the "edge deployment" claim literal. Both deliberately deferred past the core 8-week build to protect the timeline.
