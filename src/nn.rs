//! Building blocks for the YOLO26 forward graph.
//!
//! All activations and normalizations assume inference-mode constants.
//! Conv2d is implemented as a direct stride-aware loop over the input; for the
//! 1x1 case we collapse it to a matrix multiply so the inner loop is contiguous.
use crate::tensor::Tensor;

/// Convolution that follows PyTorch's NCHW convention with zero-padding.
///
/// `weight` is `[out_channels, in_channels, kH, kW]`.
/// `bias` is `[out_channels]` (already folded from BN).
/// Supports `groups` (default 1) — depthwise convs use `groups = in_channels`.
pub fn conv2d(input: &Tensor, weight: &Tensor, bias: &Tensor, stride: (usize, usize), padding: (usize, usize)) -> Tensor {
    conv2d_grouped(input, weight, bias, stride, padding, 1)
}

pub fn conv2d_grouped(input: &Tensor, weight: &Tensor, bias: &Tensor, stride: (usize, usize), padding: (usize, usize), groups: usize) -> Tensor {
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    assert_eq!(input.shape.len(), 4, "conv input must be NCHW");
    assert_eq!(weight.shape.len(), 4, "conv weight must be OIHW");
    let n = input.n() as usize;
    let c_in = input.c() as usize;
    let h_in = input.h() as usize;
    let w_in = input.w() as usize;
    let c_out = weight.shape[0] as usize;
    let c_per_group_in = weight.shape[1] as usize;
    let k_h = weight.shape[2] as usize;
    let k_w = weight.shape[3] as usize;
    assert_eq!(c_in, c_per_group_in * groups, "channel mismatch: input={} weight.in_per_group={} groups={}", c_in, c_per_group_in, groups);
    assert_eq!(c_out % groups, 0, "out_channels {} not divisible by groups {}", c_out, groups);
    let c_out_per_group = c_out / groups;
    let h_out = (h_in + 2 * ph - k_h) / sh + 1;
    let w_out = (w_in + 2 * pw - k_w) / sw + 1;
    let mut out = Tensor::zeros(vec![n as u64, c_out as u64, h_out as u64, w_out as u64]);
    let i_strides = input.strides();
    let o_strides = out.strides();

    if k_h == 1 && k_w == 1 && sh == 1 && sw == 1 {
        let spatial = h_out * w_out;
        for ni in 0..n {
            for g in 0..groups {
                let i_base_g = g * c_per_group_in;
                for o_local in 0..c_out_per_group {
                    let o = g * c_out_per_group + o_local;
                    let bias_o = bias.data[o];
                    let w_base = o * c_per_group_in;
                    for p in 0..spatial {
                        let mut sum = bias_o;
                        let in_base = ni * i_strides[0] + (i_base_g) * i_strides[1] + p;
                        for i in 0..c_per_group_in {
                            sum += weight.data[w_base + i] * input.data[in_base + i * i_strides[1]];
                        }
                        out.data[ni * o_strides[0] + o * o_strides[1] + p] = sum;
                    }
                }
            }
        }
        return out;
    }

    for ni in 0..n {
        for g in 0..groups {
            for o_local in 0..c_out_per_group {
                let o = g * c_out_per_group + o_local;
                let bias_o = bias.data[o];
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = bias_o;
                        for i in 0..c_per_group_in {
                            let ci = g * c_per_group_in + i;
                            for kh in 0..k_h {
                                let ih = oh * sh + kh;
                                if ih < ph || ih >= h_in + ph {
                                    continue;
                                }
                                let ih_idx = ih - ph;
                                for kw in 0..k_w {
                                    let iw = ow * sw + kw;
                                    if iw < pw || iw >= w_in + pw {
                                        continue;
                                    }
                                    let iw_idx = iw - pw;
                                    let in_off = ni * i_strides[0] + ci * i_strides[1] + ih_idx * i_strides[2] + iw_idx;
                                    let w_off = ((o * c_per_group_in + i) * k_h + kh) * k_w + kw;
                                    sum += input.data[in_off] * weight.data[w_off];
                                }
                            }
                        }
                        out.data[ni * o_strides[0] + o * o_strides[1] + oh * o_strides[2] + ow] = sum;
                    }
                }
            }
        }
    }
    out
}

/// Depthwise convolution: groups = in_channels (assumed in_channels == out_channels).
pub fn dw_conv2d(input: &Tensor, weight: &Tensor, bias: &Tensor, stride: (usize, usize), padding: (usize, usize)) -> Tensor {
    let c = input.c() as usize;
    conv2d_grouped(input, weight, bias, stride, padding, c)
}

/// SiLU activation in-place.
pub fn silu_inplace(t: &mut Tensor) {
    for v in t.data.iter_mut() {
        let x = *v;
        *v = x / (1.0 + (-x).exp());
    }
}

/// Fused Conv2d + SiLU (BN already absorbed into the weight/bias).
pub fn conv_silu(input: &Tensor, weight: &Tensor, bias: &Tensor, stride: (usize, usize), padding: (usize, usize)) -> Tensor {
    let mut y = conv2d(input, weight, bias, stride, padding);
    silu_inplace(&mut y);
    y
}

/// MaxPool2d (kH x kW, stride sH x sW, padding pH x pW).
pub fn max_pool2d(input: &Tensor, k: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Tensor {
    let (kh, kw) = k;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let n = input.n() as usize;
    let c = input.c() as usize;
    let h_in = input.h() as usize;
    let w_in = input.w() as usize;
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    let mut out = Tensor::zeros(vec![n as u64, c as u64, h_out as u64, w_out as u64]);
    let i_strides = input.strides();
    let o_strides = out.strides();
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut m = f32::NEG_INFINITY;
                    for kh in 0..kh {
                        let ih = oh * sh + kh;
                        if ih < ph || ih >= h_in + ph {
                            continue;
                        }
                        let ih_idx = ih - ph;
                        for kw in 0..kw {
                            let iw = ow * sw + kw;
                            if iw < pw || iw >= w_in + pw {
                                continue;
                            }
                            let iw_idx = iw - pw;
                            let v = input.data[ni * i_strides[0] + ci * i_strides[1] + ih_idx * i_strides[2] + iw_idx];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    out.data[ni * o_strides[0] + ci * o_strides[1] + oh * o_strides[2] + ow] = m;
                }
            }
        }
    }
    out
}

/// Nearest-neighbor 2x upsample on NCHW.
pub fn upsample_nearest2x(input: &Tensor) -> Tensor {
    let n = input.n() as usize;
    let c = input.c() as usize;
    let h_in = input.h() as usize;
    let w_in = input.w() as usize;
    let h_out = h_in * 2;
    let w_out = w_in * 2;
    let mut out = Tensor::zeros(vec![n as u64, c as u64, h_out as u64, w_out as u64]);
    let i_strides = input.strides();
    let o_strides = out.strides();
    for ni in 0..n {
        for ci in 0..c {
            for ih in 0..h_in {
                for iw in 0..w_in {
                    let v = input.data[ni * i_strides[0] + ci * i_strides[1] + ih * i_strides[2] + iw];
                    let base_o = ni * o_strides[0] + ci * o_strides[1] + (ih * 2) * o_strides[2] + (iw * 2);
                    out.data[base_o] = v;
                    out.data[base_o + 1] = v;
                    out.data[base_o + o_strides[2]] = v;
                    out.data[base_o + o_strides[2] + 1] = v;
                }
            }
        }
    }
    out
}

/// Concatenate a list of tensors along the channel dimension.
pub fn concat_channels(inputs: &[&Tensor]) -> Tensor {
    assert!(!inputs.is_empty(), "concat empty");
    let n = inputs[0].n();
    let h = inputs[0].h();
    let w = inputs[0].w();
    let total_c: u64 = inputs.iter().map(|t| t.c()).sum();
    let mut out = Tensor::zeros(vec![n, total_c, h, w]);
    let o_strides = out.strides();
    for t in inputs {
        assert_eq!(t.n(), n, "concat batch mismatch");
        assert_eq!(t.h(), h, "concat h mismatch");
        assert_eq!(t.w(), w, "concat w mismatch");
    }
    let mut cursor = 0usize;
    for t in inputs {
        let c = t.c() as usize;
        let i_strides = t.strides();
        for ni in 0..(n as usize) {
            for ci in 0..c {
                for hi in 0..(h as usize) {
                    let i_off = ni * i_strides[0] + ci * i_strides[1] + hi * i_strides[2];
                    let o_off = ni * o_strides[0] + (cursor + ci) * o_strides[1] + hi * o_strides[2];
                    for wi in 0..(w as usize) {
                        out.data[o_off + wi] = t.data[i_off + wi];
                    }
                }
            }
        }
        cursor += c;
    }
    out
}