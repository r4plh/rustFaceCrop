use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array4, CowArray, IxDyn};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;


use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};

const CONFIDENCE_THRESHOLD: f32 = 0.25;
const NMS_THRESHOLD: f32 = 0.45;
const PADDING: u32 = 10;
const BATCH_SIZE: usize = 64; 


#[derive(Debug, Clone, Copy, PartialEq)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

impl BoundingBox {

    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }


    fn iou(&self, other: &BoundingBox) -> f32 {
        let x_inter1 = self.x1.max(other.x1);
        let y_inter1 = self.y1.max(other.y1);
        let x_inter2 = self.x2.min(other.x2);
        let y_inter2 = self.y2.min(other.y2);

        let intersection_area =
            (x_inter2 - x_inter1).max(0.0) * (y_inter2 - y_inter1).max(0.0);
        let union_area = self.area() + other.area() - intersection_area;

        if union_area > 0.0 {
            intersection_area / union_area
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Detection {
    bbox: BoundingBox,
    confidence: f32,
}


fn non_max_suppression(
    detections: &mut Vec<Detection>,
    nms_threshold: f32,
) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }


    detections.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let mut final_detections = Vec::new();
    let mut is_suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if is_suppressed[i] {
            continue;
        }

        final_detections.push(detections[i]);

        for j in (i + 1)..detections.len() {
            if is_suppressed[j] {
                continue;
            }

            let iou = detections[i].bbox.iou(&detections[j].bbox);
            if iou > nms_threshold {
                is_suppressed[j] = true;
            }
        }
    }
    final_detections
}

fn load_local_images(dir_path: &str) -> Result<Vec<(PathBuf, DynamicImage, String)>> {
    let mut images = Vec::new();
    println!("📂 Loading images recursively from: '{}'", dir_path);

    let image_extensions = ["png", "jpg", "jpeg", "bmp"];


    fn walk_directory_recursive(
        path: &std::path::Path,
        images: &mut Vec<(PathBuf, DynamicImage, String)>,
        extensions: &[&str],
    ) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();

            if entry_path.is_dir() {

                walk_directory_recursive(&entry_path, images, extensions)?;
            } else if entry_path.is_file() {

                if let Some(ext) = entry_path.extension().and_then(|s| s.to_str()) {
                    if extensions.contains(&ext.to_lowercase().as_str()) {

                        let parent_dir = entry_path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown")
                            .to_string();
                        
                        println!("    -> Found image: {} (category: {})", 
                                entry_path.file_name().unwrap_or_default().to_string_lossy(), 
                                parent_dir);
                        
                        match image::open(&entry_path) {
                            Ok(img) => {
                                images.push((entry_path, img, parent_dir));
                            }
                            Err(e) => eprintln!(
                                "⚠️  Failed to open image {}: {}",
                                entry_path.display(),
                                e
                            ),
                        }
                    }
                }
            }
        }
        Ok(())
    }

    let base_path = std::path::Path::new(dir_path);
    walk_directory_recursive(base_path, &mut images, &image_extensions)?;

    if images.is_empty() {
        let error_msg = format!(
            "No valid images found in directory '{}'. Please ensure it contains images in subdirectories.",
            dir_path
        );
        return Err(anyhow::anyhow!(error_msg));
    } else {
        let unique_subfolders: HashSet<&String> = images.iter().map(|(_, _, subfolder)| subfolder).collect();
        println!("✅ Loaded {} images from {} categories.", images.len(), unique_subfolders.len());
    }

    Ok(images)
}

struct YOLOModel {
    session: Session,
    input_size: (usize, usize),
}

impl YOLOModel {
    fn new(model_path: &str) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("YOLO")
            .build()?
            .into_arc();

        let session = SessionBuilder::new(&environment)?
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])?
            .with_model_from_file(model_path)?;

        Ok(Self {
            session,
            input_size: (640, 640),
        })
    }

    fn preprocess_image(&self, img: &DynamicImage) -> Result<Array4<f32>> {
        let img = img.resize_exact(
            self.input_size.0 as u32,
            self.input_size.1 as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let img_rgb = img.to_rgb8();
        let (width, height) = img_rgb.dimensions();
        let mut array = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        for y in 0..height {
            for x in 0..width {
                let pixel = img_rgb.get_pixel(x, y);
                array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }
        Ok(array)
    }

    fn run_inference(&self, input: Array4<f32>) -> Result<Vec<Detection>> {
        let input_dyn = input.into_dyn();
        let cow_array: CowArray<'_, f32, IxDyn> = CowArray::from(input_dyn);
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;
        
        let outputs = self.session.run(vec![input_tensor])?;
        
        let output = outputs[0].try_extract()?;
        let view = output.view();
        let shape = view.shape();
        
        let data: Vec<f32> = view.iter().cloned().collect();
        let mut detections = Vec::new();
        
        //  correct output format: [1, 5, 8400]
        if shape.len() == 3 && shape[0] == 1 && shape[1] == 5 {
            let num_features = shape[1]; // 5
            let num_detections = shape[2]; // 8400
            
            for i in 0..num_detections {
                let x_idx = 0 * num_detections + i;
                let y_idx = 1 * num_detections + i;
                let w_idx = 2 * num_detections + i;
                let h_idx = 3 * num_detections + i;
                let conf_idx = 4 * num_detections + i;
                
                if conf_idx >= data.len() {
                    break;
                }
                
                let confidence = data[conf_idx];
                
                if confidence > CONFIDENCE_THRESHOLD {
                    let cx = data[x_idx];
                    let cy = data[y_idx];
                    let w = data[w_idx];
                    let h = data[h_idx];
                    
                    // Convert from center coordinates to corner coordinates
                    let x1 = cx - w / 2.0;
                    let y1 = cy - h / 2.0;
                    let x2 = cx + w / 2.0;
                    let y2 = cy + h / 2.0;
                    
                    // Clamp coordinates to input image bounds
                    let x1 = x1.max(0.0).min(self.input_size.0 as f32);
                    let y1 = y1.max(0.0).min(self.input_size.1 as f32);
                    let x2 = x2.max(0.0).min(self.input_size.0 as f32);
                    let y2 = y2.max(0.0).min(self.input_size.1 as f32);
                    
                    // Validate bounding box
                    if x2 > x1 && y2 > y1 && (x2 - x1) > 10.0 && (y2 - y1) > 10.0 {
                        detections.push(Detection {
                            bbox: BoundingBox { x1, y1, x2, y2 },
                            confidence,
                        });
                    }
                }
            }
        }
        

        Ok(non_max_suppression(&mut detections, NMS_THRESHOLD))
    }
}

fn main() -> Result<()> {
    println!(" YOLO Face Detection & Cropping in Rust");
    println!("==========================================");

    println!("\n Loading YOLO model...");
    let model = YOLOModel::new("yolov11n-face.onnx")?;
    println!(" Model loaded successfully!");

    println!("\n Loading local images...");
    let images = load_local_images("WIDER_val")?;
    if images.is_empty() {
        println!("\nNo images to process. Exiting.");
        return Ok(());
    }
    println!(" Loaded {} images for processing", images.len());

    println!("\n🔍 Running face detection inference in batches of {}...", BATCH_SIZE);
    println!("=====================================");

    let output_dir = "wider_face_crops";
    fs::create_dir_all(output_dir)?;

    let mut total_faces_saved = 0;
    
    // Process images in batches (for organization, but inference is still per-image)
    for (batch_idx, batch_images) in images.chunks(BATCH_SIZE).enumerate() {
        println!("\n--- Processing batch {}/{} ({} images) ---", 
                 batch_idx + 1, 
                 (images.len() + BATCH_SIZE - 1) / BATCH_SIZE,
                 batch_images.len());
        
        // Process each image in the batch individually (since model expects batch_size=1)
        for (img_idx_in_batch, (image_path, img, subfolder_name)) in batch_images.iter().enumerate() {
            let global_img_idx = batch_idx * BATCH_SIZE + img_idx_in_batch + 1;
            
            println!("\n  → Processing image {}/{}: {} (from {})", 
                     global_img_idx, images.len(), 
                     image_path.file_name().unwrap_or_default().to_string_lossy(), 
                     subfolder_name);
            
            let (original_width, original_height) = img.dimensions();
            
            // Process single image
            let input_tensor = model.preprocess_image(img)?;
            let detections = model.run_inference(input_tensor)?;
            
            println!("    → Detected {} face(s) after NMS", detections.len());
            
            let x_scale = original_width as f32 / model.input_size.0 as f32;
            let y_scale = original_height as f32 / model.input_size.1 as f32;
            
            // Extract base filename without extension for naming
            let original_filename = image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            
            for (face_idx, detection) in detections.iter().enumerate() {
                let bbox = &detection.bbox;
                
                // Scale coordinates from input size (640x640) back to original image size
                let x1 = (bbox.x1 * x_scale) as u32;
                let y1 = (bbox.y1 * y_scale) as u32;
                let x2 = (bbox.x2 * x_scale) as u32;
                let y2 = (bbox.y2 * y_scale) as u32;
                
                // Add padding and ensure coordinates are within image bounds
                let x1_pad = x1.saturating_sub(PADDING);
                let y1_pad = y1.saturating_sub(PADDING);
                let x2_pad = (x2 + PADDING).min(original_width);
                let y2_pad = (y2 + PADDING).min(original_height);
                
                let crop_width = x2_pad.saturating_sub(x1_pad);
                let crop_height = y2_pad.saturating_sub(y1_pad);
                
                if crop_width > 0 && crop_height > 0 {
                    let cropped_face = image::imageops::crop_imm(img, x1_pad, y1_pad, crop_width, crop_height);
                    
                    // NEW NAMING: subfolder_imagename_face_X.jpg
                    let output_filename = if detections.len() == 1 {
                        format!("{}_{}_face.jpg", subfolder_name, original_filename)
                    } else {
                        format!("{}_{}_face_{}.jpg", subfolder_name, original_filename, face_idx + 1)
                    };
                    let output_path = PathBuf::from(output_dir).join(&output_filename);
                    
                    cropped_face.to_image().save(&output_path)?;
                    total_faces_saved += 1;
                    println!("      → Face {}: Confidence {:.2}, Saved to: {}", 
                             face_idx + 1, detection.confidence, output_filename);
                } else {
                    println!("      → Face {}: Invalid crop dimensions ({}x{}), skipping", 
                             face_idx + 1, crop_width, crop_height);
                }
            }
        }
    }

    println!("\n WIDER FACE processing completed!");
    println!("Total faces cropped and saved: {}", total_faces_saved);
    println!("Results saved in '{}' folder", output_dir);
    println!("Ready for face recognition training! ");
    Ok(())
}