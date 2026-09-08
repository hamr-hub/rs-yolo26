//! YOLO26 Detect head + NMS.
use crate::tensor::Tensor;

/// Make per-anchor grid centers in feature-pixels (stride-aware).
///
/// Returns `(anchors_xy, strides)`:
///   anchors_xy: shape `[num_anchors, 2]`, the (x, y) center of each anchor in pixels of the input image.
///   strides:    shape `[num_anchors]`, the stride (8/16/32) for each anchor.
pub fn make_anchors(shapes: &[(usize, usize)], strides: &[usize]) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut anchors = Vec::new();
    let mut s_list = Vec::new();
    for (idx, &(h, w)) in shapes.iter().enumerate() {
        let s = strides[idx] as f32;
        for yi in 0..h {
            for xi in 0..w {
                anchors.push([(xi as f32 + 0.5) * s, (yi as f32 + 0.5) * s]);
                s_list.push(s);
            }
        }
    }
    (anchors, s_list)
}

/// Decode raw box predictions for DFL-free reg_max=1 case.
///
/// `bboxes` has shape `[num_anchors, 4]` (l, t, r, t distances) — we use the
/// plain (l, t, r, b) -> (cx - l*stride, cy - t*stride, cx + r*stride, cy + b*stride) formula
/// that is equivalent to DFL=Identity.
pub fn decode_bboxes_ltrb(bboxes: &[f32], anchors: &[[f32; 2]], strides: &[f32], out: &mut [f32]) {
    debug_assert_eq!(bboxes.len(), anchors.len() * 4);
    debug_assert_eq!(out.len(), anchors.len() * 4);
    for (i, (a, &s)) in anchors.iter().zip(strides.iter()).enumerate() {
        let l = bboxes[i * 4] * s;
        let t = bboxes[i * 4 + 1] * s;
        let r = bboxes[i * 4 + 2] * s;
        let b = bboxes[i * 4 + 3] * s;
        out[i * 4] = a[0] - l;
        out[i * 4 + 1] = a[1] - t;
        out[i * 4 + 2] = a[0] + r;
        out[i * 4 + 3] = a[1] + b;
    }
}

/// Detect head with the head structure used by yolo26n: 3 scales, each scale
/// has cv2 (box) and cv3 (cls), each is a small sequential:
///
///   cv2[i] = Conv(x, c2, 3) -> Conv(c2, c2, 3) -> Conv(c2, 4*reg_max, 1)  (no act on last)
///   cv3[i] = DWConv(x, x, 3) -> Conv(x, c3, 1) -> DWConv(c3, c3, 3) -> Conv(c3, c3, 1) -> Conv(c3, nc, 1)  (no act on last)
///
/// Where reg_max=1 → dfl=Identity → output is already 4 plain (l, t, r, b) values per anchor.
pub struct DetectHead {
    pub cv2: Vec<HeadSequential>,   // box head per scale
    pub cv3: Vec<ClsSequential>,    // cls head per scale
}

pub struct HeadSequential {
    pub c1_w: Tensor, pub c1_b: Tensor,
    pub c2_w: Tensor, pub c2_b: Tensor,
    pub c3_w: Tensor, pub c3_b: Tensor, // final 1x1, no bias for last conv usually
}

pub struct ClsSequential {
    pub dw1_w: Tensor, pub dw1_b: Tensor,
    pub pw1_w: Tensor, pub pw1_b: Tensor,
    pub dw2_w: Tensor, pub dw2_b: Tensor,
    pub pw2_w: Tensor, pub pw2_b: Tensor,
    pub c3_w: Tensor, pub c3_b: Tensor, // final 1x1
}

impl DetectHead {
    /// Run the box head and produce `[num_anchors_total, 4]` raw (l, t, r, b).
    pub fn forward_box(&self, feats: &[&Tensor]) -> Vec<f32> {
        let mut all = Vec::new();
        for (i, f) in feats.iter().enumerate() {
            let h = &self.cv2[i];
            let mut y = crate::nn::conv_silu(f, &h.c1_w, &h.c1_b, (1, 1), (1, 1));
            y = crate::nn::conv_silu(&y, &h.c2_w, &h.c2_b, (1, 1), (1, 1));
            y = crate::nn::conv2d(&y, &h.c3_w, &h.c3_b, (1, 1), (0, 0));
            // y has shape [1, 4, H, W]
            let n = y.n() as usize;
            let c = y.c() as usize;
            let hh = y.h() as usize;
            let ww = y.w() as usize;
            assert_eq!(n, 1);
            assert_eq!(c, 4);
            let s = y.strides();
            for hi in 0..hh {
                for wi in 0..ww {
                    for ci in 0..4 {
                        all.push(y.data[0 * s[0] + ci * s[1] + hi * s[2] + wi]);
                    }
                }
            }
        }
        all
    }

    /// Run the class head and produce `[num_anchors_total, nc]` raw logits.
    pub fn forward_cls(&self, feats: &[&Tensor]) -> Vec<f32> {
        let mut all = Vec::new();
        for (i, f) in feats.iter().enumerate() {
            let h = &self.cv3[i];
            let mut y = crate::nn::dw_conv2d(f, &h.dw1_w, &h.dw1_b, (1, 1), (1, 1));
            // SiLU in-place after dw conv
            for v in y.data.iter_mut() {
                let x = *v;
                *v = x / (1.0 + (-x).exp());
            }
            y = crate::nn::conv_silu(&y, &h.pw1_w, &h.pw1_b, (1, 1), (0, 0));
            y = crate::nn::dw_conv2d(&y, &h.dw2_w, &h.dw2_b, (1, 1), (1, 1));
            for v in y.data.iter_mut() {
                let x = *v;
                *v = x / (1.0 + (-x).exp());
            }
            y = crate::nn::conv_silu(&y, &h.pw2_w, &h.pw2_b, (1, 1), (0, 0));
            y = crate::nn::conv2d(&y, &h.c3_w, &h.c3_b, (1, 1), (0, 0));
            let n = y.n() as usize;
            let c = y.c() as usize;
            let hh = y.h() as usize;
            let ww = y.w() as usize;
            assert_eq!(n, 1);
            let s = y.strides();
            for hi in 0..hh {
                for wi in 0..ww {
                    for ci in 0..c {
                        all.push(y.data[0 * s[0] + ci * s[1] + hi * s[2] + wi]);
                    }
                }
            }
        }
        all
    }
}

/// Sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One-shot end-to-end Decode: box + sigmoid(scores), top-k by max-class score,
/// return a list of (x1, y1, x2, y2, conf, class_idx).
pub fn end2end_decode(
    boxes_ltrb: &[f32],
    cls_logits: &[f32],
    anchors: &[[f32; 2]],
    strides: &[f32],
    nc: usize,
    max_det: usize,
    conf_thresh: f32,
) -> Vec<(f32, f32, f32, f32, f32, u32)> {
    let n_anchors = anchors.len();
    assert_eq!(boxes_ltrb.len(), n_anchors * 4);
    assert_eq!(cls_logits.len(), n_anchors * nc);

    // Decode boxes to xyxy
    let mut boxes = vec![0f32; n_anchors * 4];
    decode_bboxes_ltrb(boxes_ltrb, anchors, strides, &mut boxes);

    // For each anchor, compute class score and top class
    let mut best_class = vec![0u32; n_anchors];
    let mut best_score = vec![0f32; n_anchors];
    for a in 0..n_anchors {
        let row = &cls_logits[a * nc..(a + 1) * nc];
        let mut best_i = 0u32;
        let mut best_s = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            let s = sigmoid(v);
            if s > best_s {
                best_s = s;
                best_i = i as u32;
            }
        }
        best_class[a] = best_i;
        best_score[a] = best_s;
    }

    // Threshold + sort
    let mut keep: Vec<(f32, u32)> = Vec::new(); // (score, anchor_idx)
    for a in 0..n_anchors {
        if best_score[a] >= conf_thresh {
            keep.push((best_score[a], a as u32));
        }
    }
    keep.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    keep.truncate(max_det);

    keep.iter()
        .map(|&(score, a)| {
            let x1 = boxes[a as usize * 4];
            let y1 = boxes[a as usize * 4 + 1];
            let x2 = boxes[a as usize * 4 + 2];
            let y2 = boxes[a as usize * 4 + 3];
            (x1, y1, x2, y2, score, best_class[a as usize])
        })
        .collect()
}

/// Class-agnostic NMS on xyxy boxes (sorted by score).
///
/// Returns the indices into `boxes` that survive NMS.
pub fn nms_class_agnostic(boxes: &[(f32, f32, f32, f32)], scores: &[f32], iou_thresh: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &order {
            if j == i || suppressed[j] {
                continue;
            }
            let iou = iou_xyxy(boxes[i], boxes[j]);
            if iou > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

#[inline]
fn iou_xyxy(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = a.2.min(b.2);
    let y2 = a.3.min(b.3);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    let inter = iw * ih;
    let area_a = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let area_b = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// NMS-based one-to-many decode: per-class NMS, returns detections.
pub fn nms_decode(
    boxes_ltrb: &[f32],
    cls_logits: &[f32],
    anchors: &[[f32; 2]],
    strides: &[f32],
    nc: usize,
    conf_thresh: f32,
    iou_thresh: f32,
    max_det: usize,
) -> Vec<(f32, f32, f32, f32, f32, u32)> {
    let n_anchors = anchors.len();
    let mut boxes = vec![0f32; n_anchors * 4];
    decode_bboxes_ltrb(boxes_ltrb, anchors, strides, &mut boxes);

    let mut out = Vec::new();
    for c in 0..nc {
        let mut class_boxes: Vec<(f32, f32, f32, f32)> = Vec::new();
        let mut class_scores: Vec<f32> = Vec::new();
        let mut class_idx: Vec<u32> = Vec::new();
        for a in 0..n_anchors {
            let s = sigmoid(cls_logits[a * nc + c]);
            if s >= conf_thresh {
                class_boxes.push((boxes[a * 4], boxes[a * 4 + 1], boxes[a * 4 + 2], boxes[a * 4 + 3]));
                class_scores.push(s);
                class_idx.push(c as u32);
            }
        }
        if class_boxes.is_empty() {
            continue;
        }
        let keep = nms_class_agnostic(&class_boxes, &class_scores, iou_thresh);
        for &k in &keep {
            out.push((class_boxes[k].0, class_boxes[k].1, class_boxes[k].2, class_boxes[k].3, class_scores[k], class_idx[k]));
            if out.len() >= max_det {
                return out;
            }
        }
    }
    out
}