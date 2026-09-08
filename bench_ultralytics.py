#!/usr/bin/env python3
"""Benchmark: ultralytics YOLO predict on a given image (CPU)."""
import sys
import time
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else '/tmp/yolo26n.pt'
image = sys.argv[2] if len(sys.argv) > 2 else '/tmp/bus.jpg'
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5

m = YOLO(weights)

# Warm-up
m(image, verbose=False)

times = []
results = None
for _ in range(iters):
    t0 = time.perf_counter()
    results = m(image, verbose=False)
    t1 = time.perf_counter()
    times.append(t1 - t0)
avg = sum(times) / len(times)
print(f'ultralytics predict: avg {avg*1000:.1f} ms over {iters} iters')
print(f'ultralytics detections:')
for r in results:
    for i in range(len(r.boxes)):
        cls = int(r.boxes.cls[i]); conf = float(r.boxes.conf[i])
        xyxy = r.boxes.xyxy[i].tolist()
        print(f'  cls={cls} conf={conf:.3f} xyxy={xyxy}')