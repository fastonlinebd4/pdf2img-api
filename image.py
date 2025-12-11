from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import uuid
import os
import aiofiles
from pdf2image import convert_from_path

app = FastAPI(title="PDF → Photo + Signature Extract API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
OUT_DIR = "output"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def extract_signature(full_image_path):
    """ Crop bottom area → auto-detect signature """
    img = cv2.imread(full_image_path)

    if img is None:
        return None

    h, w, c = img.shape

    # Crop bottom 35% area (signature থাকে নিচে)
    crop = img[int(h * 0.60):h, 0:w]

    # Convert to gray
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold for signature detection
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours (ink strokes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Get bounding box for all strokes
    x_min = min([cv2.boundingRect(c)[0] for c in contours])
    y_min = min([cv2.boundingRect(c)[1] for c in contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

    sign_crop = crop[y_min:y_max, x_min:x_max]

    sign_file = f"{uuid.uuid4()}_sign.jpg"
    sign_path = os.path.join(OUT_DIR, sign_file)

    cv2.imwrite(sign_path, sign_crop)

    return sign_file


@app.post("/convert-pdf")
async def convert_nid(pdf_file: UploadFile = File(...)):

    if not pdf_file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.pdf")

    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(await pdf_file.read())

    # Convert PDF → Image
    pages = convert_from_path(temp_path, 300)

    if len(pages) == 0:
        raise HTTPException(500, "PDF conversion failed")

    # First page = NID front page
    page_image = pages[0]
    img_id = f"{uuid.uuid4()}_full.jpg"
    full_path = os.path.join(OUT_DIR, img_id)
    page_image.save(full_path)

    # Extract signature
    sign_file = extract_signature(full_path)

    response = {
        "status": "success",
        "photo_image": f"/preview/{img_id}",
        "sign_image": f"/preview/{sign_file}" if sign_file else None
    }

    return JSONResponse(response)


@app.get("/preview/{filename}")
async def preview(filename):
    path = os.path.join(OUT_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "File not found")

    return FileResponse(path, media_type="image/jpeg")
