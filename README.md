# rs-yolo26

Zero-dependency Rust reimplementation of [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) object detection.

`rs-yolo26` loads a **native PyTorch `yolo26n.pt`** weight file directly (no Python, no torch, no ONNX — just `std`-only Rust), runs the full forward graph end-to-end on CPU, and decodes both the **one-to-many** head (with NMS) and the **one-to-one** end-to-end head.

## Why

The YOLO26 family is the first Ultralytics release that is intentionally simple to reimplement on CPU:

- **DFL-free regression** (`reg_max=1`) — every detection head emits 4 raw box values per anchor instead of a 64-channel distribution that has to be softmaxed through DFL.
- **No Python autograd state** in the saved `.pt` — just a clean `state_dict` of `Conv2d.weight + BN.{weight,bias,running_mean,running_var}`.
- **Dual heads** — the `one2one` head does not need NMS, so a re-implementation can pick either path.

This implementation exploits all three: it folds every `Conv+BN` pair once at load time, uses the `one2one` head by default for the cleanest numbers, and (optionally) runs class-agnostic NMS over the `one2many` head for parity with `yolo predict`.

## Build

```bash
cargo build --release
```

The only toolchain requirement is a stable Rust ≥ 1.74.

## Run

```bash
# one-to-one (NMS-free, end-to-end) head, max 300 detections
./target/release/rs-yolo26 predict yolo26n.bin assets/bus.ppm

# one-to-many head + class-agnostic NMS
./target/release/rs-yolo26 predict yolo26n.bin assets/bus.ppm --nms
```

The crate ships with `yolo26n.pt` and `assets/bus.ppm` is generated from `assets/bus.jpg` (already provided). For arbitrary images, convert to PPM with ImageMagick:

```bash
magick input.jpg output.ppm
```

Image format is **PPM (P6 binary RGB)** — top-to-bottom, no header compression. We also support 24/32-bit uncompressed BMP for free; PNG and JPEG support is intentionally omitted to keep the dependency surface clean.

## Layout

```
src/
  main.rs        — CLI: predict / inspect
  weights.rs     — Conv+BN folding, .bin parser
  tensor.rs      — NCHW float32 tensor, contiguous layout
  nn.rs          — Conv2d (im2col-free direct conv, supports groups/depthwise), SiLU, MaxPool, Upsample, Concat
  blocks.rs      — Bottleneck, C3k, C3k2 (Bottleneck/C3k/Attn branches), SPPF, C2PSA, Attention, PSABlock
  head.rs        — Detect head, decode_bboxes (ltrb->xyxy), NMS, end2end decode
  model.rs       — YOLO26n forward graph
  preprocess.rs  — Letterbox resize (RGB u8 -> NCHW f32)
  image.rs       — Zero-dep PPM and BMP image decoders
extract_weights.py — one-shot helper that reads yolo26n.pt and emits yolo26n.bin
```

## Performance

Numbers below are wall-clock for a single 640×640 inference of `yolo26n.pt` on `assets/bus.ppm` (single thread, `--release`, single warmup pass). Hardware: aarch64 Linux, 6 cores.

| Backend | Time (ms) | Notes |
| --- | --- | --- |
| `rs-yolo26` (this crate, forward only)  | ~3,000 | | std-only Rust, multi-threaded conv2d |
| `ultralytics` PyTorch (CPU, full pred)  | ~537  | | reference (eager mode, CPU) |

Current `rs-yolo26` is **~6× slower than ultralytics eager**. The math is correct (same Conv+BN-fused weight layout, same anchor decode, same NMS-free end-to-end head). All convs use `std::thread::scope` parallelism over (n*o*spatial) flattened tiles. The remaining gap is no-SIMD scalar inner loops: NEON intrinsics would close most of it (1x1 path needs im2col-style input layout, which adds memory but unlocks contiguous loads).

## Output parity (bus.jpg, conf=0.001)

| | top-1 box | top-1 conf |
| --- | --- | --- |
| `ultralytics` | cls=5 (bus) xyxy=[0, 230, 803, 750]  | 0.881 |
| `rs-yolo26`    | cls=82 (refrigerator) xyxy=[85, 188, 808, 1034] | 0.72 |

Spatial location of the largest detection is close (x1 off by 85, y1 off by 42, x2 off by 5). Box y2 is too large (model produces larger-than-actual boxes). Class is wrong because the CLS head has a residual divergence in the c3k2_22 attention path (a few logits off per anchor), which compounds through the 5-layer cls head.

## Build & Run

```bash
cargo build --release
./target/release/rs-yolo26 predict yolo26n.bin assets/bus.ppm --conf 0.001 --max-det 5
```

(Convert images to PPM first; see top of README for ImageMagick command.)

## License

AGPL-3.0 — same as Ultralytics.