#!/usr/bin/env python3
"""Extract YOLO26 weights into a single .bin for rs-yolo26 (Conv+BN folded).

Requires `pip install torch ultralytics`. Run this once per weights file:
    python3 extract_weights.py yolo26n.pt yolo26n.bin

The output is a flat binary; load it with `WeightStore::load`.
"""
import os
import struct
import sys
import torch


def fuse_conv_bn(conv_w, bn_rm, bn_rv, bn_w, bn_b, bn_eps=1e-5):
    w = conv_w.detach().float()
    rm = bn_rm.detach().float()
    rv = bn_rv.detach().float()
    bw = bn_w.detach().float()
    bb = bn_b.detach().float()
    scale = bw / torch.sqrt(rv + bn_eps)
    fused_w = w * scale.view(-1, 1, 1, 1)
    fused_b = bb - rm * scale
    return fused_w.contiguous(), fused_b.contiguous()


def main(pt_path, out_path):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"].float().state_dict()

    # Walk the state_dict, fuse each {prefix}.conv + {prefix}.bn pair into a single Conv.
    # Bare convs (no bn) and BN-less bias layers pass through.
    keys = list(sd.keys())
    seen = set()
    fused = {}  # name -> (weight_tensor, bias_tensor) or None for non-conv params

    for k in keys:
        if k in seen:
            continue
        if k.endswith(".conv.weight"):
            prefix = k[: -len(".conv.weight")]
            bn_w_k = f"{prefix}.bn.weight"
            bn_b_k = f"{prefix}.bn.bias"
            bn_rm_k = f"{prefix}.bn.running_mean"
            bn_rv_k = f"{prefix}.bn.running_var"
            if all(x in sd for x in (bn_w_k, bn_b_k, bn_rm_k, bn_rv_k)):
                fw, fb = fuse_conv_bn(
                    sd[k],
                    sd[bn_rm_k],
                    sd[bn_rv_k],
                    sd[bn_w_k],
                    sd[bn_b_k],
                )
                fused[f"{prefix}.conv.weight"] = fw
                fused[f"{prefix}.conv.bias"] = fb
                seen.update([k, bn_w_k, bn_b_k, bn_rm_k, bn_rv_k])
                continue
        # Bare conv or other parameter
        if k.endswith(".num_batches_tracked") or k.endswith(".running_mean") or k.endswith(".running_var"):
            seen.add(k)
            continue
        if k.endswith(".bn.weight") or k.endswith(".bn.bias"):
            # Could be a "bias" only path — emit zero bias
            prefix = k[: -len(".weight" if k.endswith(".weight") else ".bias")]
            if f"{prefix}.weight" in sd:
                fused[k] = sd[k].float().contiguous()
                seen.add(k)
            continue
        fused[k] = sd[k].float().contiguous()
        seen.add(k)

    print(f"Total fused parameters: {len(fused)}")

    # Write the .bin
    with open(out_path, "wb") as f:
        # Magic + version + index_off placeholder + count
        f.write(b"Y26W")
        f.write(struct.pack("<B", 1))               # version
        f.write(struct.pack("<I", 0))               # index_off placeholder
        f.write(struct.pack("<I", len(fused)))      # count
        offsets = []
        for name, tensor in fused.items():
            offsets.append(f.tell())
            arr = tensor.numpy()
            f.write(arr.tobytes())
        index_pos = f.tell()
        # Backpatch index offset (header offset = 4 + 1 = 5, write u32)
        f.seek(5)
        f.write(struct.pack("<I", index_pos))
        # Write index
        f.seek(index_pos)
        f.write(struct.pack("<I", len(fused)))
        for (name, tensor), off in zip(fused.items(), offsets):
            name_b = name.encode("utf-8")
            shape = list(tensor.shape)
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<q", int(d)))
            f.write(struct.pack("<QQ", off, tensor.numel() * tensor.element_size()))
    print(f"Wrote {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolo26n.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/yolo26n.bin"
    main(pt_path, out_path)