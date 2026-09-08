//! YOLO26n forward graph (after Conv+BN folding).
use crate::blocks::{Attention, Bottleneck, C3k, C3k2, C2PSA, C3k2_C3k, C3k2_Attn, C3k2AttnBlock, PSABlock, SPPF};
use crate::head::{DetectHead, HeadSequential, ClsSequential};
use crate::nn::{conv_silu, conv2d, upsample_nearest2x, concat_channels};
use crate::tensor::Tensor;
use crate::weights::WeightStore;

fn load_bottleneck(store: &WeightStore, prefix: &str, shortcut: bool) -> Bottleneck {
    let key = format!("{prefix}.cv1.conv.weight");
    let w1 = store.get(&key).unwrap_or_else(|| panic!("missing {} (have {} keys)", key, store.tensors.len()));
    let b1 = store.get(&format!("{prefix}.cv1.conv.bias")).expect("bottleneck cv1 bias");
    let w2 = store.get(&format!("{prefix}.cv2.conv.weight")).expect("bottleneck cv2 weight");
    let b2 = store.get(&format!("{prefix}.cv2.conv.bias")).expect("bottleneck cv2 bias");
    let k1 = w1.shape[2] as usize;
    let k2 = w2.shape[2] as usize;
    Bottleneck { cv1_w: w1.clone(), cv1_b: b1.clone(), cv2_w: w2.clone(), cv2_b: b2.clone(), k1, k2, add: shortcut }
}

fn load_c3k(store: &WeightStore, prefix: &str, n: usize) -> C3k {
    let cv1_w = store.get(&format!("{prefix}.cv1.conv.weight")).expect("c3k cv1 w").clone();
    let cv1_b = store.get(&format!("{prefix}.cv1.conv.bias")).expect("c3k cv1 b").clone();
    let cv2_w = store.get(&format!("{prefix}.cv2.conv.weight")).expect("c3k cv2 w").clone();
    let cv2_b = store.get(&format!("{prefix}.cv2.conv.bias")).expect("c3k cv2 b").clone();
    let cv3_w = store.get(&format!("{prefix}.cv3.conv.weight")).expect("c3k cv3 w").clone();
    let cv3_b = store.get(&format!("{prefix}.cv3.conv.bias")).expect("c3k cv3 b").clone();
    let mut blocks = Vec::with_capacity(n);
    for i in 0..n {
        blocks.push(load_bottleneck(store, &format!("{prefix}.m.{i}"), true));
    }
    C3k { cv1_w, cv1_b, cv2_w, cv2_b, cv3_w, cv3_b, blocks }
}

fn load_c3k2_bottleneck(store: &WeightStore, prefix: &str, n: usize) -> C3k2 {
    let cv1_w = store.get(&format!("{prefix}.cv1.conv.weight")).expect("c3k2 cv1 weight").clone();
    let cv1_b = store.get(&format!("{prefix}.cv1.conv.bias")).expect("c3k2 cv1 bias").clone();
    let cv2_w = store.get(&format!("{prefix}.cv2.conv.weight")).expect("c3k2 cv2 weight").clone();
    let cv2_b = store.get(&format!("{prefix}.cv2.conv.bias")).expect("c3k2 cv2 bias").clone();
    let mut blocks = Vec::with_capacity(n);
    for i in 0..n {
        let cv1_out_c = cv1_w.shape[0] as usize;
        let c_block = cv1_out_c / 2;
        blocks.push(load_bottleneck(store, &format!("{prefix}.m.{i}"), true));
    }
    C3k2 { cv1_w, cv1_b, cv2_w, cv2_b, blocks }
}

fn load_c3k2_c3k(store: &WeightStore, prefix: &str, n_outer: usize, n_inner: usize) -> C3k2_C3k {
    let cv1_w = store.get(&format!("{prefix}.cv1.conv.weight")).expect("c3k2 cv1 weight").clone();
    let cv1_b = store.get(&format!("{prefix}.cv1.conv.bias")).expect("c3k2 cv1 bias").clone();
    let cv2_w = store.get(&format!("{prefix}.cv2.conv.weight")).expect("c3k2 cv2 weight").clone();
    let cv2_b = store.get(&format!("{prefix}.cv2.conv.bias")).expect("c3k2 cv2 bias").clone();
    let mut blocks = Vec::with_capacity(n_outer);
    for i in 0..n_outer {
        blocks.push(load_c3k(store, &format!("{prefix}.m.{i}"), n_inner));
    }
    C3k2_C3k { cv1_w, cv1_b, cv2_w, cv2_b, blocks }
}

fn load_attention(store: &WeightStore, prefix: &str) -> Attention {
    let qkv_w = store.get(&format!("{prefix}.qkv.conv.weight")).expect("attn qkv weight");
    let qkv_b = store.get(&format!("{prefix}.qkv.conv.bias")).expect("attn qkv bias");
    let proj_w = store.get(&format!("{prefix}.proj.conv.weight")).expect("attn proj weight");
    let proj_b = store.get(&format!("{prefix}.proj.conv.bias")).expect("attn proj bias");
    let pe_w = store.get(&format!("{prefix}.pe.conv.weight")).expect("attn pe weight");
    let pe_b = store.get(&format!("{prefix}.pe.conv.bias")).expect("attn pe bias");
    let c = qkv_w.shape[0] as usize;
    let c_in = qkv_w.shape[1] as usize;
    let mut nh = 2usize;
    let mut kd = 32usize;
    let mut hd = 64usize;
    if c == 2 * c_in {
        hd = 64;
        nh = c_in / hd;
        if nh < 1 {
            nh = 1;
            hd = c_in;
        }
        kd = hd / 2;
    }
    Attention {
        qkv_w: qkv_w.clone(), qkv_b: qkv_b.clone(),
        proj_w: proj_w.clone(), proj_b: proj_b.clone(),
        pe_w: pe_w.clone(), pe_b: pe_b.clone(),
        num_heads: nh, head_dim: hd, key_dim: kd,
    }
}

fn load_psablock(store: &WeightStore, prefix: &str) -> PSABlock {
    let attn = load_attention(store, &format!("{prefix}.attn"));
    PSABlock {
        attn,
        ffn1_w: store.get(&format!("{prefix}.ffn.0.conv.weight")).expect("ffn1 w").clone(),
        ffn1_b: store.get(&format!("{prefix}.ffn.0.conv.bias")).expect("ffn1 b").clone(),
        ffn2_w: store.get(&format!("{prefix}.ffn.1.conv.weight")).expect("ffn2 w").clone(),
        ffn2_b: store.get(&format!("{prefix}.ffn.1.conv.bias")).expect("ffn2 b").clone(),
    }
}

fn load_c3k2_attn(store: &WeightStore, prefix: &str, n: usize) -> C3k2_Attn {
    let cv1_w = store.get(&format!("{prefix}.cv1.conv.weight")).expect("c3k2a cv1 w").clone();
    let cv1_b = store.get(&format!("{prefix}.cv1.conv.bias")).expect("c3k2a cv1 b").clone();
    let cv2_w = store.get(&format!("{prefix}.cv2.conv.weight")).expect("c3k2a cv2 w").clone();
    let cv2_b = store.get(&format!("{prefix}.cv2.conv.bias")).expect("c3k2a cv2 b").clone();
    let mut blocks = Vec::with_capacity(n);
    for i in 0..n {
        let bottleneck = load_bottleneck(store, &format!("{prefix}.m.{i}.0"), true);
        let psa = load_psablock(store, &format!("{prefix}.m.{i}.1"));
        blocks.push(C3k2AttnBlock { bottleneck, psa });
    }
    C3k2_Attn { cv1_w, cv1_b, cv2_w, cv2_b, blocks }
}

fn load_sppf(store: &WeightStore, prefix: &str) -> SPPF {
    SPPF {
        cv1_w: store.get(&format!("{prefix}.cv1.conv.weight")).expect("sppf cv1 weight").clone(),
        cv1_b: store.get(&format!("{prefix}.cv1.conv.bias")).expect("sppf cv1 bias").clone(),
        cv2_w: store.get(&format!("{prefix}.cv2.conv.weight")).expect("sppf cv2 weight").clone(),
        cv2_b: store.get(&format!("{prefix}.cv2.conv.bias")).expect("sppf cv2 bias").clone(),
        k: 5,
        n: 3,
    }
}

fn load_c2psa(store: &WeightStore, prefix: &str, n: usize) -> C2PSA {
    let mut blocks = Vec::with_capacity(n);
    for i in 0..n {
        blocks.push(load_psablock(store, &format!("{prefix}.m.{i}")));
    }
    C2PSA {
        cv1_w: store.get(&format!("{prefix}.cv1.conv.weight")).expect("c2psa cv1 w").clone(),
        cv1_b: store.get(&format!("{prefix}.cv1.conv.bias")).expect("c2psa cv1 b").clone(),
        cv2_w: store.get(&format!("{prefix}.cv2.conv.weight")).expect("c2psa cv2 w").clone(),
        cv2_b: store.get(&format!("{prefix}.cv2.conv.bias")).expect("c2psa cv2 b").clone(),
        blocks,
    }
}

fn load_head(store: &WeightStore, prefix: &str) -> DetectHead {
    let mut cv2 = Vec::new();
    let mut cv3 = Vec::new();
    for i in 0..3 {
        let p = format!("{prefix}.cv2.{i}");
        cv2.push(HeadSequential {
            c1_w: store.get(&format!("{p}.0.conv.weight")).expect("cv2.0 w").clone(),
            c1_b: store.get(&format!("{p}.0.conv.bias")).expect("cv2.0 b").clone(),
            c2_w: store.get(&format!("{p}.1.conv.weight")).expect("cv2.1 w").clone(),
            c2_b: store.get(&format!("{p}.1.conv.bias")).expect("cv2.1 b").clone(),
            c3_w: store.get(&format!("{p}.2.weight")).expect("cv2.2 w").clone(),
            c3_b: store.get(&format!("{p}.2.bias")).expect("cv2.2 b").clone(),
        });
        let p = format!("{prefix}.cv3.{i}");
        cv3.push(ClsSequential {
            dw1_w: store.get(&format!("{p}.0.0.conv.weight")).expect("cv3 dw1 w").clone(),
            dw1_b: store.get(&format!("{p}.0.0.conv.bias")).expect("cv3 dw1 b").clone(),
            pw1_w: store.get(&format!("{p}.0.1.conv.weight")).expect("cv3 pw1 w").clone(),
            pw1_b: store.get(&format!("{p}.0.1.conv.bias")).expect("cv3 pw1 b").clone(),
            dw2_w: store.get(&format!("{p}.1.0.conv.weight")).expect("cv3 dw2 w").clone(),
            dw2_b: store.get(&format!("{p}.1.0.conv.bias")).expect("cv3 dw2 b").clone(),
            pw2_w: store.get(&format!("{p}.1.1.conv.weight")).expect("cv3 pw2 w").clone(),
            pw2_b: store.get(&format!("{p}.1.1.conv.bias")).expect("cv3 pw2 b").clone(),
            c3_w: store.get(&format!("{p}.2.weight")).expect("cv3 c3 w").clone(),
            c3_b: store.get(&format!("{p}.2.bias")).expect("cv3 c3 b").clone(),
        });
    }
    DetectHead { cv2, cv3 }
}

pub struct Yolo26n {
    pub nc: usize,
    pub conv0: (Tensor, Tensor),
    pub conv1: (Tensor, Tensor),
    pub c3k2_2: C3k2,
    pub conv3: (Tensor, Tensor),
    pub c3k2_4: C3k2,
    pub conv5: (Tensor, Tensor),
    pub c3k2_6: C3k2_C3k,
    pub conv7: (Tensor, Tensor),
    pub c3k2_8: C3k2_C3k,
    pub sppf: SPPF,
    pub c2psa: C2PSA,
    pub c3k2_13: C3k2_C3k,
    pub c3k2_16: C3k2_C3k,
    pub conv17: (Tensor, Tensor),
    pub c3k2_19: C3k2_C3k,
    pub conv20: (Tensor, Tensor),
    pub c3k2_22: C3k2_Attn,
    pub head: DetectHead,
}

impl Yolo26n {
    pub fn load(store: &WeightStore) -> Self {
        Yolo26n {
            nc: 80,
            conv0: (
                store.get("model.0.conv.weight").expect("conv0 w").clone(),
                store.get("model.0.conv.bias").expect("conv0 b").clone(),
            ),
            conv1: (
                store.get("model.1.conv.weight").expect("conv1 w").clone(),
                store.get("model.1.conv.bias").expect("conv1 b").clone(),
            ),
            c3k2_2: load_c3k2_bottleneck(store, "model.2", 1),
            conv3: (
                store.get("model.3.conv.weight").expect("conv3 w").clone(),
                store.get("model.3.conv.bias").expect("conv3 b").clone(),
            ),
            c3k2_4: load_c3k2_bottleneck(store, "model.4", 1),
            conv5: (
                store.get("model.5.conv.weight").expect("conv5 w").clone(),
                store.get("model.5.conv.bias").expect("conv5 b").clone(),
            ),
            c3k2_6: load_c3k2_c3k(store, "model.6", 1, 2),
            conv7: (
                store.get("model.7.conv.weight").expect("conv7 w").clone(),
                store.get("model.7.conv.bias").expect("conv7 b").clone(),
            ),
            c3k2_8: load_c3k2_c3k(store, "model.8", 1, 2),
            sppf: load_sppf(store, "model.9"),
            c2psa: load_c2psa(store, "model.10", 1),
            c3k2_13: load_c3k2_c3k(store, "model.13", 1, 2),
            c3k2_16: load_c3k2_c3k(store, "model.16", 1, 2),
            conv17: (
                store.get("model.17.conv.weight").expect("conv17 w").clone(),
                store.get("model.17.conv.bias").expect("conv17 b").clone(),
            ),
            c3k2_19: load_c3k2_c3k(store, "model.19", 1, 2),
            conv20: (
                store.get("model.20.conv.weight").expect("conv20 w").clone(),
                store.get("model.20.conv.bias").expect("conv20 b").clone(),
            ),
            c3k2_22: load_c3k2_attn(store, "model.22", 1),
            head: load_head(store, "model.23"),
        }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let mut x = conv_silu(x, &self.conv0.0, &self.conv0.1, (2, 2), (1, 1));
        x = conv_silu(&x, &self.conv1.0, &self.conv1.1, (2, 2), (1, 1));
        let b2 = self.c3k2_2.forward(&x);
        x = conv_silu(&b2, &self.conv3.0, &self.conv3.1, (2, 2), (1, 1));
        let b4 = self.c3k2_4.forward(&x);
        x = conv_silu(&b4, &self.conv5.0, &self.conv5.1, (2, 2), (1, 1));
        let b6 = self.c3k2_6.forward(&x);
        x = conv_silu(&b6, &self.conv7.0, &self.conv7.1, (2, 2), (1, 1));
        let _b8 = self.c3k2_8.forward(&x);
        x = self.sppf.forward(&_b8);
        let p5 = self.c2psa.forward(&x);

        let u = upsample_nearest2x(&p5);
        let c13_in = concat_channels(&[&u, &b6]);
        let c13 = self.c3k2_13.forward(&c13_in);
        let u2 = upsample_nearest2x(&c13);
        let c16_in = concat_channels(&[&u2, &b4]);
        let c16 = self.c3k2_16.forward(&c16_in);
        let c17 = conv_silu(&c16, &self.conv17.0, &self.conv17.1, (2, 2), (1, 1));
        let c18_in = concat_channels(&[&c17, &c13]);
        let c19 = self.c3k2_19.forward(&c18_in);
        let c20 = conv_silu(&c19, &self.conv20.0, &self.conv20.1, (2, 2), (1, 1));
        let c21_in = concat_channels(&[&c20, &p5]);
        let c22 = self.c3k2_22.forward(&c21_in);
        (c16, c19, c22)
    }
}