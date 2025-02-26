import os
import torch
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

ocr_model = VisionEncoderDecoderModel.from_pretrained("models/trocr-finetuned-model")
processor = TrOCRProcessor.from_pretrained("models/trocr-finetuned-model")
ocr_model.eval()




# Load YOLO detection model
det_model = YOLO('models/lic_det.pt')
det_model.eval()

# Define helper functions

def detect_license_plate(image):
    original_shape = image.shape[:2]  # (height, width)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image to (640, 640)
    resized = cv2.resize(image_rgb, (640, 640))
    # Convert to tensor: shape becomes (1, 3, 640, 640)
    input_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    
    start_time = time.time()
    results = det_model(input_tensor)
    det_time = time.time() - start_time
    return results, det_time, original_shape



def crop_from_detections(image, results, original_shape):
    """
    Extracts a bounding box from YOLO detection results and scales it back to the original image.
    Returns the cropped region and the box coordinates [x1, y1, x2, y2].
    """
    boxes = results[0].boxes.xyxy 
    if boxes is None or boxes.shape[0] == 0:
        return None, None
    box = boxes[0]
    box = box.int().tolist()
    x1, y1, x2, y2 = box

    # Get original image dimensions
    orig_h, orig_w = original_shape
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    # Scale coordinates to original image dimensions
    x1 = int(x1 * scale_x)
    x2 = int(x2 * scale_x)
    y1 = int(y1 * scale_y)
    y2 = int(y2 * scale_y)

    cropped = image[y1:y2, x1:x2]
    return cropped, [x1, y1, x2, y2]


def perform_ocr(cropped_image):
    """
    Preprocesses the cropped license plate image, passes it to the OCR model,
    and decodes the generated output using beam search.
    """
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    target_size = (384, 384)
    resized = cv2.resize(image_rgb, target_size)

    resized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension: shape becomes (1, 384, 384, 3)
    input_array = np.expand_dims(resized, axis=0)
    
    # Convert to a torch tensor and permute dimensions to (batch, channels, height, width)
    input_tensor = torch.from_numpy(input_array).permute(0, 3, 1, 2)
    
    start_time = time.time()
    with torch.no_grad():
        generated_ids = ocr_model.generate(
            input_tensor,
            max_length=32, 
            num_beams=4,
            early_stopping=True
        )
    ocr_time = time.time() - start_time
    predicted_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return predicted_text, ocr_time

def pipeline(image_path):

     # Check if the input is already an image array
    if isinstance(image_path, np.ndarray):
        image = image_path
    elif isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
    else:
        raise ValueError("Unsupported input type. Expected a file path or numpy array.")
    
    # Run license plate detection; get original shape as well.
    detections, det_time, original_shape = detect_license_plate(image)
    cropped, box = crop_from_detections(image, detections, original_shape)
    if cropped is None:
        print("No license plate detected.")
        return
    
    # Run OCR on the cropped license plate image
    ocr_result, ocr_time = perform_ocr(cropped)
    #print("OCR result:", ocr_result)
    
    
    return ocr_result    



if __name__ == "__main__":
    folder_path = "test"
    for image in os.listdir(folder_path):
        pipeline(image_path=os.path.join(folder_path, image))
