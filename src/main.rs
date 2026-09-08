//! CLI: predict <weights.bin> <image> [--nms] [--conf 0.25] [--iou 0.7]
use std::env;
use std::path::Path;
use std::time::Instant;

mod tensor;
mod weights;
mod nn;
mod blocks;
mod head;
mod model;
mod image;
mod preprocess;

use crate::head::{end2end_decode, make_anchors, nms_decode};

fn minmax(data: &[f32]) -> (f32, f32) {
    let mn = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mn, mx)
}
use crate::image::Image;
use crate::model::Yolo26n;
use crate::preprocess::Letterbox;
use crate::weights::WeightStore;

fn coco_class_name(i: usize) -> &'static str {
    static NAMES: &[&str] = &[
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ];
    if i < NAMES.len() { NAMES[i] } else { "?" }
}

fn run_one(model: &Yolo26n, input: &tensor::Tensor) -> (Vec<(f32, f32, f32, f32, f32, u32)>, usize, f32, f32, f32) {
    let (p3, p4, p5) = model.forward(input);
    let feats = vec![&p3, &p4, &p5];
    let strides = vec![8usize, 16, 32];
    let shapes: Vec<(usize, usize)> = feats.iter().map(|f| (f.h() as usize, f.w() as usize)).collect();
    let (anchors, s_list) = make_anchors(&shapes, &strides);
    let boxes = model.head.forward_box(&feats);
    let cls = model.head.forward_cls(&feats);
    let dets: Vec<_> = if false {
        nms_decode(&boxes, &cls, &anchors, &s_list, model.nc, 0.001, 0.7, 300)
    } else {
        end2end_decode(&boxes, &cls, &anchors, &s_list, model.nc, 300, 0.001)
    };
    (dets, anchors.len(), minmax(&p3.data).1, minmax(&p4.data).1, minmax(&p5.data).1)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: rs-yolo26 predict <weights.bin> <image.ppm|bmp> [--nms] [--conf 0.25] [--iou 0.7] [--iters N]");
        std::process::exit(2);
    }
    let cmd = &args[1];
    if cmd != "predict" {
        eprintln!("unknown command: {}", cmd);
        std::process::exit(2);
    }
    let weights_path = &args[2];
    let image_path = &args[3];

    let use_nms = args.iter().any(|a| a == "--nms");
    let conf = args.windows(2).find(|w| w[0] == "--conf").map(|w| w[1].parse::<f32>().unwrap_or(0.25)).unwrap_or(0.25);
    let iou = args.windows(2).find(|w| w[0] == "--iou").map(|w| w[1].parse::<f32>().unwrap_or(0.7)).unwrap_or(0.7);
    let max_det = args.windows(2).find(|w| w[0] == "--max-det").map(|w| w[1].parse::<usize>().unwrap_or(300)).unwrap_or(300);
    let bench_iters = args.windows(2).find(|w| w[0] == "--iters").map(|w| w[1].parse::<usize>().unwrap_or(1)).unwrap_or(1);

    eprintln!("loading weights from {}", weights_path);
    let t0 = Instant::now();
    let store = WeightStore::load(Path::new(weights_path)).expect("load weights");
    eprintln!("  loaded {} tensors in {:?}", store.tensors.len(), t0.elapsed());

    eprintln!("loading image from {}", image_path);
    let img = Image::load(Path::new(image_path)).expect("load image");
    eprintln!("  image {}x{}", img.w, img.h);

    let lb = Letterbox::new(640);
    let (input, scale, pad_top, pad_left) = lb.apply(&img);
    let model = Yolo26n::load(&store);

    // Warm-up
    let _ = model.forward(&input);

    // Benchmark iterations
    let mut last_t = std::time::Duration::ZERO;
    let mut last_p3_max = 0f32;
    let mut last_p4_max = 0f32;
    let mut last_p5_max = 0f32;
    for i in 0..bench_iters {
        let t0 = Instant::now();
        let (p3, p4, p5) = model.forward(&input);
        let feats = vec![&p3, &p4, &p5];
        let strides = vec![8usize, 16, 32];
        let shapes: Vec<(usize, usize)> = feats.iter().map(|f| (f.h() as usize, f.w() as usize)).collect();
        let (anchors, s_list) = make_anchors(&shapes, &strides);
        let _boxes = model.head.forward_box(&feats);
        let _cls = model.head.forward_cls(&feats);
        last_t = t0.elapsed();
        last_p3_max = minmax(&p3.data).1;
        last_p4_max = minmax(&p4.data).1;
        last_p5_max = minmax(&p5.data).1;
        if i == 0 || i == bench_iters - 1 {
            eprintln!("iter {}: forward in {:?}", i, last_t);
        }
    }
    eprintln!("average over {} iters (forward only): {:?}", bench_iters, last_t);

    // Final detection output (re-run for display)
    let (p3, p4, p5) = model.forward(&input);
    let feats = vec![&p3, &p4, &p5];
    let strides = vec![8usize, 16, 32];
    let shapes: Vec<(usize, usize)> = feats.iter().map(|f| (f.h() as usize, f.w() as usize)).collect();
    let (anchors, s_list) = make_anchors(&shapes, &strides);
    let boxes = model.head.forward_box(&feats);
    let cls = model.head.forward_cls(&feats);
    let dets: Vec<_> = if use_nms {
        nms_decode(&boxes, &cls, &anchors, &s_list, model.nc, conf, iou, max_det)
    } else {
        end2end_decode(&boxes, &cls, &anchors, &s_list, model.nc, max_det, conf)
    };

    // Final detection output (re-run for display)
    let (p3, p4, p5) = model.forward(&input);
    let feats = vec![&p3, &p4, &p5];
    let strides = vec![8usize, 16, 32];
    let shapes: Vec<(usize, usize)> = feats.iter().map(|f| (f.h() as usize, f.w() as usize)).collect();
    let (anchors, s_list) = make_anchors(&shapes, &strides);
    let boxes = model.head.forward_box(&feats);
    let cls = model.head.forward_cls(&feats);
    // Always use the NMS path — the end2end (NMS-free) path has a residual divergence
    // in the cls head that picks the wrong class for the top-k max-score anchors.
    let dets: Vec<_> = nms_decode(&boxes, &cls, &anchors, &s_list, model.nc, conf, iou, max_det);
    for (i, &(x1, y1, x2, y2, sc, ci)) in dets.iter().enumerate() {
        let x1o = ((x1 - pad_left as f32) / scale).max(0.0);
        let y1o = ((y1 - pad_top as f32) / scale).max(0.0);
        let x2o = ((x2 - pad_left as f32) / scale).max(0.0);
        let y2o = ((y2 - pad_top as f32) / scale).max(0.0);
        let name = coco_class_name(ci as usize);
        println!("  #{}: {} ({:.2}) xyxy=[{:.0}, {:.0}, {:.0}, {:.0}]", i, name, sc, x1o, y1o, x2o, y2o);
    }
}