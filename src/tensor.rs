//! Tensor type: NCHW float32 with contiguous layout.

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<u64>, // [N, C, H, W] expected
}

impl Tensor {
    pub fn zeros(shape: Vec<u64>) -> Self {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        Self { data: vec![0.0; n], shape }
    }

    pub fn ones(shape: Vec<u64>) -> Self {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        Self { data: vec![1.0; n], shape }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<u64>) -> Self {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        assert_eq!(data.len(), n, "data len {} != shape product {}", data.len(), n);
        Self { data, shape }
    }

    pub fn from_bytes(bytes: &[u8], shape: Vec<u64>) -> Self {
        assert_eq!(bytes.len() % 4, 0, "tensor bytes not aligned to f32");
        let n = bytes.len() / 4;
        let mut data = vec![0f32; n];
        // safe: bytes are little-endian (we wrote them that way)
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn n(&self) -> u64 { self.shape.get(0).copied().unwrap_or(1) }
    pub fn c(&self) -> u64 { self.shape.get(1).copied().unwrap_or(1) }
    pub fn h(&self) -> u64 { self.shape.get(2).copied().unwrap_or(1) }
    pub fn w(&self) -> u64 { self.shape.get(3).copied().unwrap_or(1) }

    /// Strides for [N, C, H, W] contiguous layout: [C*H*W, H*W, W, 1].
    pub fn strides(&self) -> [usize; 4] {
        let h = self.h() as usize;
        let w = self.w() as usize;
        let c = self.c() as usize;
        [c * h * w, h * w, w, 1]
    }

    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        let s = self.strides();
        self.data[n * s[0] + c * s[1] + h * s[2] + w * s[3]]
    }

    pub fn set(&mut self, n: usize, c: usize, h: usize, w: usize, v: f32) {
        let s = self.strides();
        self.data[n * s[0] + c * s[1] + h * s[2] + w * s[3]] = v;
    }
}