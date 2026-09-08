//! Higher-level blocks used by YOLO26n.
use crate::nn::{conv_silu, conv2d, max_pool2d};
use crate::tensor::Tensor;

/// Bottleneck: conv1 (k0) -> conv2 (k1) [+ input if shortcut && c1==c2], then SiLU.
pub struct Bottleneck {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub k1: usize,
    pub k2: usize,
    pub add: bool,
}

impl Bottleneck {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Both conv1 and conv2 use autopad = k/2 (Ultralytics default).
        let p1 = self.k1 / 2;
        let p2 = self.k2 / 2;
        let mut y = conv_silu(x, &self.cv1_w, &self.cv1_b, (1, 1), (p1, p1));
        y = conv2d(&y, &self.cv2_w, &self.cv2_b, (1, 1), (p2, p2));
        if self.add {
            assert_eq!(y.shape, x.shape, "Bottleneck add shape mismatch: y={:?} x={:?}", y.shape, x.shape);
            for (yi, xi) in y.data.iter_mut().zip(x.data.iter()) {
                *yi += *xi;
            }
        }
        for v in y.data.iter_mut() {
            let x = *v;
            *v = x / (1.0 + (-x).exp());
        }
        y
    }
}

/// C3 block used by C3k (with custom kernel): cv1 -> m(x) and cv2(x) -> cat -> cv3.
pub struct C3k {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub cv3_w: Tensor,
    pub cv3_b: Tensor,
    pub blocks: Vec<Bottleneck>,
}

impl C3k {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // cv1 + SiLU
        let cv1_out = conv_silu(x, &self.cv1_w, &self.cv1_b, (1, 1), (0, 0));
        // cv2 + SiLU
        let cv2_out = conv_silu(x, &self.cv2_w, &self.cv2_b, (1, 1), (0, 0));
        // m(cv1_out) chain
        let mut cur = cv1_out;
        for blk in &self.blocks {
            cur = blk.forward(&cur);
        }
        // cat(cur, cv2_out) -> cv3 + SiLU
        let cat = crate::nn::concat_channels(&[&cur, &cv2_out]);
        conv_silu(&cat, &self.cv3_w, &self.cv3_b, (1, 1), (0, 0))
    }
}

/// C3k2 with c3k=False: cv1 -> split -> [m_0, m_1, ...] -> cat -> cv2.
pub struct C3k2 {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub blocks: Vec<Bottleneck>,
}

impl C3k2 {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let cv1_out = conv_silu(x, &self.cv1_w, &self.cv1_b, (1, 1), (0, 0));
        let c_ = cv1_out.c() / 2;
        let n = cv1_out.n() as usize;
        let h = cv1_out.h() as usize;
        let w = cv1_out.w() as usize;
        let i_strides = cv1_out.strides();
        let total_c = c_ * (2 + self.blocks.len() as u64);
        let mut cat = Tensor::zeros(vec![cv1_out.n(), total_c, cv1_out.h(), cv1_out.w()]);
        let o_strides = cat.strides();

        // Copy left part
        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        cat.data[ni * o_strides[0] + ci * o_strides[1] + hi * o_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + ci * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }

        // Extract right
        let mut right = Tensor::zeros(vec![cv1_out.n(), c_, cv1_out.h(), cv1_out.w()]);
        let r_strides = right.strides();
        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        right.data[ni * r_strides[0] + ci * r_strides[1] + hi * r_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + (ci + c_ as usize) * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }

        // Place right at chunk offset c_
        {
            let chunk_off = c_ as usize;
            let r_strides = right.strides();
            for ni in 0..n {
                for hi in 0..h {
                    for ci in 0..(c_ as usize) {
                        for wi in 0..w {
                            cat.data[ni * o_strides[0] + (chunk_off + ci) * o_strides[1] + hi * o_strides[2] + wi] =
                                right.data[ni * r_strides[0] + ci * r_strides[1] + hi * r_strides[2] + wi];
                        }
                    }
                }
            }
        }

        let mut cur = right;
        for (i, blk) in self.blocks.iter().enumerate() {
            let out = blk.forward(&cur);
            let chunk_off = c_ as usize * (2 + i);
            let o_strides_local = out.strides();
            for ni in 0..n {
                for hi in 0..h {
                    for ci in 0..(c_ as usize) {
                        for wi in 0..w {
                            cat.data[ni * o_strides[0] + (chunk_off + ci) * o_strides[1] + hi * o_strides[2] + wi] =
                                out.data[ni * o_strides_local[0] + ci * o_strides_local[1] + hi * o_strides_local[2] + wi];
                        }
                    }
                }
            }
            cur = out;
        }

        conv_silu(&cat, &self.cv2_w, &self.cv2_b, (1, 1), (0, 0))
    }
}

/// C3k2 with c3k=True: ModuleList of C3k blocks.
pub struct C3k2_C3k {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub blocks: Vec<C3k>,
}

impl C3k2_C3k {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let cv1_out = conv_silu(x, &self.cv1_w, &self.cv1_b, (1, 1), (0, 0));
        let c_ = cv1_out.c() / 2;
        let n = cv1_out.n() as usize;
        let h = cv1_out.h() as usize;
        let w = cv1_out.w() as usize;
        let i_strides = cv1_out.strides();
        let total_c = c_ * (2 + self.blocks.len() as u64);
        let mut cat = Tensor::zeros(vec![cv1_out.n(), total_c, cv1_out.h(), cv1_out.w()]);
        let o_strides = cat.strides();

        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        cat.data[ni * o_strides[0] + ci * o_strides[1] + hi * o_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + ci * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }

        let mut right = Tensor::zeros(vec![cv1_out.n(), c_, cv1_out.h(), cv1_out.w()]);
        let r_strides = right.strides();
        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        right.data[ni * r_strides[0] + ci * r_strides[1] + hi * r_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + (ci + c_ as usize) * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }

        // Place right at chunk offset c_
        {
            let chunk_off = c_ as usize;
            let r_strides = right.strides();
            for ni in 0..n {
                for hi in 0..h {
                    for ci in 0..(c_ as usize) {
                        for wi in 0..w {
                            cat.data[ni * o_strides[0] + (chunk_off + ci) * o_strides[1] + hi * o_strides[2] + wi] =
                                right.data[ni * r_strides[0] + ci * r_strides[1] + hi * r_strides[2] + wi];
                        }
                    }
                }
            }
        }

        let mut cur = right;
        for (i, blk) in self.blocks.iter().enumerate() {
            let out = blk.forward(&cur);
            let chunk_off = c_ as usize * (2 + i);
            let o_strides_local = out.strides();
            for ni in 0..n {
                for hi in 0..h {
                    for ci in 0..(c_ as usize) {
                        for wi in 0..w {
                            cat.data[ni * o_strides[0] + (chunk_off + ci) * o_strides[1] + hi * o_strides[2] + wi] =
                                out.data[ni * o_strides_local[0] + ci * o_strides_local[1] + hi * o_strides_local[2] + wi];
                        }
                    }
                }
            }
            cur = out;
        }

        conv_silu(&cat, &self.cv2_w, &self.cv2_b, (1, 1), (0, 0))
    }
}

/// C3k2 with attn=True: ModuleList of Sequential[Bottleneck, PSABlock].
pub struct C3k2_Attn {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub blocks: Vec<C3k2AttnBlock>,
}

pub struct C3k2AttnBlock {
    pub bottleneck: Bottleneck,
    pub psa: PSABlock,
}

impl C3k2_Attn {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let cv1_out = conv_silu(x, &self.cv1_w, &self.cv1_b, (1, 1), (0, 0));
        let c_ = cv1_out.c() / 2;
        let n = cv1_out.n() as usize;
        let h = cv1_out.h() as usize;
        let w = cv1_out.w() as usize;
        let i_strides = cv1_out.strides();
        let total_c = c_ * (2 + self.blocks.len() as u64);
        let mut cat = Tensor::zeros(vec![cv1_out.n(), total_c, cv1_out.h(), cv1_out.w()]);
        let o_strides = cat.strides();

        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        cat.data[ni * o_strides[0] + ci * o_strides[1] + hi * o_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + ci * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }

        let mut right = Tensor::zeros(vec![cv1_out.n(), c_, cv1_out.h(), cv1_out.w()]);
        let r_strides = right.strides();
        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        right.data[ni * r_strides[0] + ci * r_strides[1] + hi * r_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + (ci + c_ as usize) * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }
        // Place right at chunk offset c_
        {
            let chunk_off = c_ as usize;
            let r_strides = right.strides();
            for ni in 0..n {
                for hi in 0..h {
                    for ci in 0..(c_ as usize) {
                        for wi in 0..w {
                            cat.data[ni * o_strides[0] + (chunk_off + ci) * o_strides[1] + hi * o_strides[2] + wi] =
                                right.data[ni * r_strides[0] + ci * r_strides[1] + hi * r_strides[2] + wi];
                        }
                    }
                }
            }
        }

        let mut cur = right;
        for (i, blk) in self.blocks.iter().enumerate() {
            let mut out = blk.bottleneck.forward(&cur);
            out = blk.psa.forward(&out);
            let chunk_off = c_ as usize * (2 + i);
            let o_strides_local = out.strides();
            for ni in 0..n {
                for hi in 0..h {
                    for ci in 0..(c_ as usize) {
                        for wi in 0..w {
                            cat.data[ni * o_strides[0] + (chunk_off + ci) * o_strides[1] + hi * o_strides[2] + wi] =
                                out.data[ni * o_strides_local[0] + ci * o_strides_local[1] + hi * o_strides_local[2] + wi];
                        }
                    }
                }
            }
            cur = out;
        }

        conv_silu(&cat, &self.cv2_w, &self.cv2_b, (1, 1), (0, 0))
    }
}

/// SPPF: cv1 -> [maxpool(iter)] -> concat -> cv2.
pub struct SPPF {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub k: usize,
    pub n: usize,
}

impl SPPF {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let c1 = conv2d(x, &self.cv1_w, &self.cv1_b, (1, 1), (0, 0));
        let mut y_list: Vec<Tensor> = vec![c1];
        for _ in 0..self.n {
            let prev = y_list.last().unwrap();
            let pooled = max_pool2d(prev, (self.k, self.k), (1, 1), (self.k / 2, self.k / 2));
            y_list.push(pooled);
        }
        let refs: Vec<&Tensor> = y_list.iter().collect();
        let cat = crate::nn::concat_channels(&refs);
        conv_silu(&cat, &self.cv2_w, &self.cv2_b, (1, 1), (0, 0))
    }
}

/// Position-Sensitive Attention block.
pub struct Attention {
    pub qkv_w: Tensor,
    pub qkv_b: Tensor,
    pub proj_w: Tensor,
    pub proj_b: Tensor,
    pub pe_w: Tensor,
    pub pe_b: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
    pub key_dim: usize,
}

impl Attention {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.n() as usize;
        let c = x.c() as usize;
        let h = x.h() as usize;
        let w = x.w() as usize;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let kd = self.key_dim;
        let qkv = conv2d(x, &self.qkv_w, &self.qkv_b, (1, 1), (0, 0));
        let qkv_c = qkv.c() as usize;

        let mut q = vec![0f32; n * nh * kd * h * w];
        let mut k_ = vec![0f32; n * nh * kd * h * w];
        let mut v = vec![0f32; n * nh * hd * h * w];
        // QKV channel layout per spatial position: [q_h0 (kd), k_h0 (kd), v_h0 (hd), q_h1, k_h1, v_h1, ...]
        // For each head, channels are interleaved.
        let per_head = 2 * kd + hd; // total channels per head
        let qkv_s = qkv.strides();
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    for hi_idx in 0..nh {
                        let base = ni * qkv_s[0] + hi_idx * per_head * qkv_s[1] + hi * qkv_s[2] + wi;
                        let qv_off = ((ni * nh + hi_idx) * h * w + hi * w + wi) * kd;
                        let kv_off = ((ni * nh + hi_idx) * h * w + hi * w + wi) * hd;
                        for k in 0..kd {
                            q[qv_off + k] = qkv.data[base + k * qkv_s[1]];
                            k_[qv_off + k] = qkv.data[base + (kd + k) * qkv_s[1]];
                        }
                        for k in 0..hd {
                            v[kv_off + k] = qkv.data[base + (2 * kd + k) * qkv_s[1]];
                        }
                    }
                }
            }
        }

        let spatial = h * w;
        let scale = (kd as f32).powf(-0.5);
        let mut attn_out = vec![0f32; n * nh * hd * spatial];
        for ni in 0..n {
            for hi_idx in 0..nh {
                let q_base = ((ni * nh + hi_idx) * spatial) * kd;
                let k_base = q_base;
                let v_base = ((ni * nh + hi_idx) * spatial) * hd;
                let mut scores = vec![0f32; spatial * spatial];
                for s1 in 0..spatial {
                    for s2 in 0..spatial {
                        let mut dot = 0.0f32;
                        for kk in 0..kd {
                            dot += q[q_base + s1 * kd + kk] * k_[k_base + s2 * kd + kk];
                        }
                        scores[s1 * spatial + s2] = dot * scale;
                    }
                }
                for s1 in 0..spatial {
                    let row = &mut scores[s1 * spatial..(s1 + 1) * spatial];
                    let mut max_v = f32::NEG_INFINITY;
                    for &v in row.iter() {
                        if v > max_v {
                            max_v = v;
                        }
                    }
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_v).exp();
                        sum += *v;
                    }
                    let inv = 1.0 / sum;
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                }
                for s1 in 0..spatial {
                    for hd_i in 0..hd {
                        let mut dot = 0.0f32;
                        for s2 in 0..spatial {
                            dot += scores[s1 * spatial + s2] * v[v_base + s2 * hd + hd_i];
                        }
                        attn_out[(ni * nh + hi_idx) * spatial * hd + s1 * hd + hd_i] = dot;
                    }
                }
            }
        }

        let mut o = vec![0f32; n * c * h * w];
        for ni in 0..n {
            for hi_idx in 0..nh {
                for s in 0..spatial {
                    let ih = s / w;
                    let iw = s % w;
                    for hd_i in 0..hd {
                        o[ni * c * h * w + (hi_idx * hd + hd_i) * h * w + ih * w + iw] =
                            attn_out[(ni * nh + hi_idx) * spatial * hd + s * hd + hd_i];
                    }
                }
            }
        }

        let mut v_nchw = vec![0f32; n * c * h * w];
        for ni in 0..n {
            for hi_idx in 0..nh {
                for s in 0..spatial {
                    let ih = s / w;
                    let iw = s % w;
                    for hd_i in 0..hd {
                        v_nchw[ni * c * h * w + (hi_idx * hd + hd_i) * h * w + ih * w + iw] =
                            v[(ni * nh + hi_idx) * spatial * hd + s * hd + hd_i];
                    }
                }
            }
        }
        let v_t = Tensor::from_vec(v_nchw, vec![n as u64, c as u64, h as u64, w as u64]);
        let pe = crate::nn::dw_conv2d(&v_t, &self.pe_w, &self.pe_b, (1, 1), (1, 1));
        // Ultralytics: x = attn_out + pe(v), NOT x + attn_out + pe(v). The PSABlock adds the residual.
        let mut x2 = vec![0f32; n * c * h * w];
        for i in 0..(n * c * h * w) {
            x2[i] = o[i] + pe.data[i];
        }
        let x2_t = Tensor::from_vec(x2, x.shape.clone());
        conv2d(&x2_t, &self.proj_w, &self.proj_b, (1, 1), (0, 0))
    }
}

/// PSABlock = Attention + FFN with residual.
pub struct PSABlock {
    pub attn: Attention,
    pub ffn1_w: Tensor,
    pub ffn1_b: Tensor,
    pub ffn2_w: Tensor,
    pub ffn2_b: Tensor,
}

impl PSABlock {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let a = self.attn.forward(x);
        let mut s = vec![0f32; x.data.len()];
        for i in 0..s.len() {
            s[i] = x.data[i] + a.data[i];
        }
        let s_t = Tensor::from_vec(s, x.shape.clone());
        let f1 = conv_silu(&s_t, &self.ffn1_w, &self.ffn1_b, (1, 1), (0, 0));
        let f2 = conv2d(&f1, &self.ffn2_w, &self.ffn2_b, (1, 1), (0, 0));
        let mut out = vec![0f32; x.data.len()];
        for i in 0..out.len() {
            out[i] = s_t.data[i] + f2.data[i];
        }
        Tensor::from_vec(out, x.shape.clone())
    }
}

/// C2PSA: cv1 -> split -> [m_0(b), m_1(b), ...] -> cv2 with attention.
pub struct C2PSA {
    pub cv1_w: Tensor,
    pub cv1_b: Tensor,
    pub cv2_w: Tensor,
    pub cv2_b: Tensor,
    pub blocks: Vec<PSABlock>,
}

impl C2PSA {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let cv1_out = conv_silu(x, &self.cv1_w, &self.cv1_b, (1, 1), (0, 0));
        let c_ = cv1_out.c() / 2;
        let n = cv1_out.n() as usize;
        let h = cv1_out.h() as usize;
        let w = cv1_out.w() as usize;
        let mut a = Tensor::zeros(vec![cv1_out.n(), c_, cv1_out.h(), cv1_out.w()]);
        let mut b = Tensor::zeros(vec![cv1_out.n(), c_, cv1_out.h(), cv1_out.w()]);
        let i_strides = cv1_out.strides();
        let a_strides = a.strides();
        let b_strides = b.strides();
        for ni in 0..n {
            for hi in 0..h {
                for ci in 0..(c_ as usize) {
                    for wi in 0..w {
                        a.data[ni * a_strides[0] + ci * a_strides[1] + hi * a_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + ci * i_strides[1] + hi * i_strides[2] + wi];
                        b.data[ni * b_strides[0] + ci * b_strides[1] + hi * b_strides[2] + wi] =
                            cv1_out.data[ni * i_strides[0] + (ci + c_ as usize) * i_strides[1] + hi * i_strides[2] + wi];
                    }
                }
            }
        }
        let mut cur = b;
        for blk in &self.blocks {
            cur = blk.forward(&cur);
        }
        let cat = crate::nn::concat_channels(&[&a, &cur]);
        conv_silu(&cat, &self.cv2_w, &self.cv2_b, (1, 1), (0, 0))
    }
}