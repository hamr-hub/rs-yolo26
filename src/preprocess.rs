//! Letterbox preprocessing: HWC u8 RGB -> NCHW f32 normalized.
//!
//! Reproduces Ultralytics' LetterBox: scale the longer side to fit `target`,
//! pad the rest with a constant color (default 114), no aspect-ratio crop,
//! then divide by 256 by default? Actually ultralytics divides by 255.
//! So pixel / 255.
use crate::image::Image;
use crate::tensor::Tensor;

pub struct Letterbox {
    pub target: usize,
    pub color: u8,
}

impl Letterbox {
    pub fn new(target: usize) -> Self {
        Self { target, color: 114 }
    }

    /// Returns (NCHW float32 tensor, scale, pad_top, pad_left).
    pub fn apply(&self, img: &Image) -> (Tensor, f32, usize, usize) {
        let w = img.w;
        let h = img.h;
        let r = (self.target as f32 / w as f32).min(self.target as f32 / h as f32);
        let new_w = ((w as f32 * r).round() as usize).min(self.target);
        let new_h = ((h as f32 * r).round() as usize).min(self.target);
        let pad_w = self.target - new_w;
        let pad_h = self.target - new_h;
        let pad_left = pad_w / 2;
        let pad_top = pad_h / 2;
        let pad_val = self.color as f32 / 255.0;
        let mut data = vec![pad_val; 3 * self.target * self.target];

        // Resize using nearest-neighbor for v1 (acceptable; bilinear is more accurate)
        let x_ratio = w as f32 / new_w as f32;
        let y_ratio = h as f32 / new_h as f32;
        for ny in 0..new_h {
            let sy = ((ny as f32 + 0.5) * y_ratio) as usize;
            let sy = sy.min(h - 1);
            for nx in 0..new_w {
                let sx = ((nx as f32 + 0.5) * x_ratio) as usize;
                let sx = sx.min(w - 1);
                let src = (sy * w + sx) * 3;
                let dst_y = pad_top + ny;
                let dst_x = pad_left + nx;
                let dst = (dst_y * self.target + dst_x) * 1;
                let r = img.rgb[src] as f32 / 255.0;
                let g = img.rgb[src + 1] as f32 / 255.0;
                let b = img.rgb[src + 2] as f32 / 255.0;
                data[dst] = r;
                data[self.target * self.target + dst] = g;
                data[2 * self.target * self.target + dst] = b;
            }
        }

        let t = Tensor::from_vec(data, vec![1, 3, self.target as u64, self.target as u64]);
        (t, r, pad_top, pad_left)
    }
}