//! .bin weight loader: parses the fused-weights blob produced by extract_weights.py.
//!
//! Binary format (little-endian):
//!   magic[4] = "Y26W"
//   index_offset: u64
//!   count: u32
//!   <tensor data blobs follow in order>
//!   <index follows at index_offset>:
//!     count: u32
//!     for each tensor:
//!       name_len: u32
//!       name: [u8; name_len]
//!       ndim: u32
//!       dims: [i64; ndim]
//!       offset: u64
//!       length: u64
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::tensor::Tensor;

#[derive(Debug)]
pub struct WeightStore {
    pub tensors: HashMap<String, Tensor>,
}

impl WeightStore {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::parse(&buf)
    }

    pub fn parse(buf: &[u8]) -> std::io::Result<Self> {
        // Header: magic[4] + version[1] + index_offset[u32] + count[u32]
        if buf.len() < 4 + 1 + 4 + 4 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "header too short"));
        }
        if &buf[0..4] != b"Y26W" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad magic {:?}", &buf[0..4]),
            ));
        }
        let version = buf[4];
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported version {}", version),
            ));
        }
        let index_offset = u32::from_le_bytes(buf[5..9].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(buf[9..13].try_into().unwrap()) as usize;

        // Parse index
        let mut pos = index_offset;
        let idx_count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_count, count, "index count mismatch");
        pos += 4;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let name_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let name = std::str::from_utf8(&buf[pos..pos + name_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?
                .to_string();
            pos += name_len;
            let ndim = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                let d = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as u64;
                pos += 8;
                shape.push(d);
            }
            let off = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let len = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let bytes = &buf[off..off + len];
            tensors.insert(name, Tensor::from_bytes(bytes, shape));
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
}