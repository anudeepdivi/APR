import os
import torch
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from apr_digits import CRNN
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgH = 32  # target height for OCR preprocessing
ocr_model = torch.load("APR_OCR_complete.pth", map_location=device)
ocr_model.eval()  # set model to evaluation mode

# --------------------------
# Load the pre-trained models
# --------------------------
# Load YOLO detection model
det_model = YOLO('lic_det.pt')
det_model.eval()


# --------------------------
# Define helper functions
# --------------------------
def detect_license_plate(image):
    # Save original dimensions
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
    boxes = results[0].boxes.xyxy  # coordinates relative to 640x640
    if boxes is None or boxes.shape[0] == 0:
        return None, None
    # Take the first detection
    box = boxes[0]
    # Convert to integers in the resized space
    box = box.int().tolist()
    x1, y1, x2, y2 = box

    # Compute scale factors from 640 to original dimensions
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


def ocr_inference(model, image, device, target_height=32):
    """
    Given a grayscale image (numpy array), preprocess and run it through the model,
    then decode predictions using greedy CTC decoding.
    """
    # Image is expected to be grayscale (1 channel)
    h, w = image.shape
    scale = target_height / float(h)
    new_width = int(w * scale)
    # Resize image to target height while preserving aspect ratio
    image_proc = cv2.resize(image, (new_width, target_height))
    image_proc = image_proc.astype(np.float32) / 255.0  # Normalize to [0,1]
    # Add batch and channel dimensions -> shape: (1, 1, H, W)
    image_proc = np.expand_dims(image_proc, axis=0)
    image_proc = np.expand_dims(image_proc, axis=0)
    image_tensor = torch.tensor(image_proc, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(image_tensor)  # Output shape: (T, 1, n_classes)
        y_pred = F.log_softmax(y_pred, dim=2)
        y_pred = y_pred.cpu()
        preds = y_pred.argmax(dim=2)  # Shape: (T, 1)
        preds = preds.squeeze(1)      # Shape: (T,)
    
    # Greedy CTC decoding: remove duplicates and blank tokens (blank token=0)
    pred_indices = []
    prev = None
    for idx in preds:
        idx_val = idx.item()
        if idx_val != prev and idx_val != 0:
            pred_indices.append(idx_val)
        prev = idx_val
    # Map indices to characters (assuming 1-indexed mapping)
    num_to_char = {i+1: char for i, char in enumerate("0123456789T")}
    result_text = "".join([num_to_char[i] for i in pred_indices if i in num_to_char])
    return result_text


# --------------------------
# Define the pipeline
# --------------------------
def pipeline(image_path):
    # Read the input image in BGR
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return
    
    # Run license plate detection on a resized image (640x640) and get original dimensions.
    detections, det_time, original_shape = detect_license_plate(image)
    cropped, box = crop_from_detections(image, detections, original_shape)
    if cropped is None:
        print("No license plate detected.")
        return
    
    # Convert the cropped region from BGR to grayscale for OCR inference
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Run OCR on the cropped grayscale license plate image
    ocr_result = ocr_inference(ocr_model, cropped_gray, device, target_height=imgH)
    
    print("Detection Inference Time: {:.4f} s".format(det_time))
    print("OCR Result:", ocr_result)
    
    # Visualization
    image_vis = image.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
    plt.title("License Plate Detection")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title(f"Cropped Plate\nOCR: {ocr_result}")
    plt.axis("off")
    
    plt.show()


# --------------------------
# Run the pipeline if executed directly
# --------------------------
if __name__ == "__main__":
    # Replace 'your_image.jpg' with the path to your test image
    folder_path = "test"
    for image in os.listdir(folder_path):
        pipeline(image_path=os.path.join(folder_path, image))
