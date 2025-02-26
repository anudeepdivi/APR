from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from typing import List
from app.pipeline import pipeline  # your pipeline function

app = FastAPI()
templates = Jinja2Templates(directory="app")

# HTML Frontend Endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("templates.html", {"request": request})

# HTML Upload Endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_html(request: Request, files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        result = pipeline(image)
        results.append(result)
    return templates.TemplateResponse("templates.html", {"request": request, "results": results})

# JSON API Endpoint
@app.post("/api/upload")
async def upload_api(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        result = pipeline(image)
        results.append(result)
    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
