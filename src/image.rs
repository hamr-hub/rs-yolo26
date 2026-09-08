//! Zero-dep image decoders for PPM (P6 binary RGB) and BMP (24/32-bit uncompressed).
//!
//! PNG and JPEG decoding are intentionally not implemented to keep the dependency
//! surface clean — convert with `magick input.png ppm:output.ppm` or
//! `ffmpeg -i input.png -f image2pipe -pix_fmt rgb24 output.ppm`.
use std::fs;
use std::io::{self, Read, Seek};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    /// RGB u8 contiguous, row-major: pixel (x, y) = data[(y * w + x) * 3 + c]
    pub rgb: Vec<u8>,
}

impl Image {
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut head = [0u8; 16];
        f.read_exact(&mut head)?;
        f.seek(io::SeekFrom::Start(0))?;
        if &head[0..3] == b"P6\n" || &head[0..3] == b"P5\n" || &head[0..2] == b"P6" || &head[0..2] == b"P5" {
            return load_ppm(&mut f);
        }
        if &head[0..2] == b"BM" {
            return load_bmp(&mut f);
        }
        if &head[0..8] == b"\x89PNG\r\n\x1a\n" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "PNG decoding not implemented; convert to PPM with `magick input.png ppm:output.ppm`",
            ));
        }
        if &head[0..3] == &[0xFF, 0xD8, 0xFF] {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "JPEG decoding not implemented; convert to PPM with `magick input.jpg ppm:output.ppm`",
            ));
        }
        Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported image format"))
    }
}

fn load_ppm(f: &mut fs::File) -> io::Result<Image> {
    // Read header byte-by-byte; header is whitespace + ASCII digits, terminated by single whitespace before pixel data.
    let mut header = Vec::new();
    let mut state = 0u8; // 0 = magic, 1 = dims, 2 = maxval, 3 = past header
    let mut dims = (0usize, 0usize);
    let mut maxval = 0u32;
    let mut tok = String::new();
    loop {
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte)?;
        let c = byte[0];
        header.push(c);
        match state {
            0 => {
                if c == b'\n' || c == b' ' || c == b'\t' || c == b'\r' {
                    if tok.as_bytes() != b"P6" {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("not P6 header: {:?}", tok)));
                    }
                    tok.clear();
                    state = 1;
                } else {
                    tok.push(c as char);
                }
            }
            1 => {
                if c == b'#' {
                    // comment, skip until newline
                    while c != b'\n' {
                        f.read_exact(&mut byte)?;
                        let c2 = byte[0];
                        header.push(c2);
                        if c2 == b'\n' { break; }
                    }
                    continue;
                }
                if c == b'\n' || c == b' ' || c == b'\t' || c == b'\r' {
                    if !tok.is_empty() {
                        let v: usize = tok.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad dim"))?;
                        if dims.0 == 0 { dims.0 = v; } else { dims.1 = v; }
                        tok.clear();
                        if dims.1 != 0 {
                            state = 2;
                        }
                    }
                } else {
                    tok.push(c as char);
                }
            }
            2 => {
                if c == b'\n' || c == b' ' || c == b'\t' || c == b'\r' {
                    if !tok.is_empty() {
                        maxval = tok.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad maxval"))?;
                        tok.clear();
                        state = 3;
                        break;
                    }
                } else {
                    tok.push(c as char);
                }
            }
            _ => break,
        }
    }
    if state != 3 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "header incomplete"));
    }
    if maxval > 255 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "maxval > 255 unsupported"));
    }
    let mut bin = Vec::new();
    f.read_to_end(&mut bin)?;
    let n = dims.0 * dims.1 * 3;
    if bin.len() < n {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "PPM truncated"));
    }
    Ok(Image { w: dims.0, h: dims.1, rgb: bin[..n].to_vec() })
}

fn load_bmp(f: &mut fs::File) -> io::Result<Image> {
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() < 54 || &buf[0..2] != b"BM" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad BMP"));
    }
    let data_offset = u32::from_le_bytes(buf[10..14].try_into().unwrap()) as usize;
    let dib_size = u32::from_le_bytes(buf[14..18].try_into().unwrap()) as usize;
    if dib_size < 12 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad DIB header"));
    }
    let w = i32::from_le_bytes(buf[18..22].try_into().unwrap());
    let h_signed = i32::from_le_bytes(buf[22..26].try_into().unwrap());
    let bpp = u16::from_le_bytes(buf[28..30].try_into().unwrap()) as usize;
    let compression = u32::from_le_bytes(buf[30..34].try_into().unwrap());
    if compression != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "compressed BMP unsupported"));
    }
    if bpp != 24 && bpp != 32 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("BMP bpp={} unsupported", bpp)));
    }
    let w = w.unsigned_abs() as usize;
    let flip = h_signed > 0;
    let h = h_signed.unsigned_abs() as usize;
    let bytes_per_row = (w * bpp / 8 + 3) & !3;
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        let src_y = if flip { h - 1 - y } else { y };
        let src = data_offset + src_y * bytes_per_row;
        for x in 0..w {
            let b = buf[src + x * (bpp / 8)];
            let g = buf[src + x * (bpp / 8) + 1];
            let r = buf[src + x * (bpp / 8) + 2];
            let dst = (y * w + x) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }
    Ok(Image { w, h, rgb })
}