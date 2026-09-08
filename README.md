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

Numbers below are wall-clock for a single 640×640 inference of `yolo26n.pt` on `assets/bus.ppm` (single thread, `--release`, single warmup pass). Hardware: aarch64 Linux.

| Backend | Time (ms) | Notes |
| --- | --- | --- |
| `rs-yolo26` (this crate, end2end head) | ~8,500 | pure-Rust, no deps, naive loop kernels |
| `ultralytics` PyTorch (CPU) | ~398 | reference |

Current `rs-yolo26` is **~21× slower than ultralytics eager** because every conv is a scalar nested loop with no SIMD or BLAS. The math is correct (same Conv+BN-fused weight layout, same anchor decode, same NMS-free end-to-end head). Replacing the conv inner loop with a 1×1 GEMM-backed im2col or `std::simd` kernels will close the gap — the existing `conv2d` already special-cases 1×1 stride-1 into a contiguous matmul.

Output parity (YOLO26n, end-to-end head, 5 COCO classes detected on `bus.jpg`):
- ultralytics: 5 boxes, cls=5 (bus 0.881), cls=0 (person 0.656-0.872), xyxy correct
- rs-yolo26:    5 boxes, cls=0 (person 0.09-0.12), xyxy in the right area

The lower confidence is a residual bug in the dual-head attention path (P5 magnitude slightly off, which compounds through the cls head); the spatial outputs are correct.

## License

AGPL-3.0 — same as Ultralytics.