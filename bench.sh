#!/usr/bin/env bash
# Compare rs-yolo26 against ultralytics on the same image.
set -euo pipefail

WEIGHTS="${1:-/tmp/yolo26n.pt}"
BIN="${2:-/tmp/yolo26n.bin}"
IMAGE="${3:-/tmp/bus.jpg}"
PPM="${5:-/tmp/bus.ppm}"
ITERS="${4:-5}"

if [[ ! -f "$BIN" ]]; then
  echo "Missing $BIN. Building via extract_weights.py..."
  python3 extract_weights.py "$WEIGHTS" "$BIN"
fi

if [[ ! -f "$PPM" ]]; then
  python3 -c "from PIL import Image; Image.open('$IMAGE').save('$PPM')"
fi

# Make sure rust binary is built
cargo build --release --quiet

echo
echo "== rs-yolo26 =="
./target/release/rs-yolo26 predict "$BIN" "$PPM" --iters "$ITERS" 2>&1 | tail -5

echo
echo "== ultralytics (pip install ultralytics torch --no-deps) =="
if python3 -c "import ultralytics" 2>/dev/null; then
  python3 bench_ultralytics.py "$WEIGHTS" "$IMAGE" "$ITERS"
else
  echo "ultralytics not installed; pip install ultralytics --no-deps to enable benchmark"
fi