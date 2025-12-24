from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import uuid
import aiofiles

app = FastAPI(title="PDF to Image Converter API (FREE)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_files"
OUTPUT_DIR = "output_images"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"status": "running", "mode": "100% FREE (No API)"}


@app.post("/convert-pdf")
async def convert_pdf(pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    pdf_id = str(uuid.uuid4())
    pdf_path = os.path.join(TEMP_DIR, f"{pdf_id}.pdf")

    async with aiofiles.open(pdf_path, "wb") as f:
        await f.write(await pdf_file.read())

    images = []

    try:
        doc = fitz.open(pdf_path)

        for page_number in range(len(doc)):
            page = doc[page_number]
            pix = page.get_pixmap(dpi=200)

            image_name = f"{pdf_id}_page_{page_number + 1}.png"
            image_path = os.path.join(OUTPUT_DIR, image_name)

            pix.save(image_path)

            images.append({
                "page": page_number + 1,
                "filename": image_name,
                "preview_url": f"/preview-image/{image_name}",
                "download_url": f"/download-image/{image_name}",
                "size_kb": round(os.path.getsize(image_path) / 1024, 2)
            })

        doc.close()

        return JSONResponse({
            "status": "success",
            "total_pages": len(images),
            "images": images
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@app.get("/download-image/{filename}")
async def download_image(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    async def stream():
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        stream(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/preview-image/{filename}")
async def preview_image(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    async def stream():
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(stream(), media_type="image/png")
