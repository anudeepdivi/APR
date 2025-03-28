import gdown
import zipfile
import os

det_model_id = "1cSCRwgGJAunB5M1DYUfyIEJ7pSkbwNU-"
ocr_model_id = "1Wv5IwDbwyFqFPTaxbnUUzVWGtNVgeW8c"


os.makedirs("models", exist_ok=True)
det_model_path = "models/lic_det.pt"
ocr_model_path = "models/trocr_model.zip"

if not os.path.exists(det_model_path):
    gdown.download(f"https://drive.google.com/uc?id={det_model_id}", det_model_path, quiet=False)

if not os.path.exists("models/trocr-finetuned-model/"):
    gdown.download(f"https://drive.google.com/uc?id={ocr_model_id}", ocr_model_path, quiet=False)
    with zipfile.ZipFile(ocr_model_path, "r") as zip_ref:
        zip_ref.extractall("models/")
    os.remove(ocr_model_path) 

print("Models downloaded successfully!")
