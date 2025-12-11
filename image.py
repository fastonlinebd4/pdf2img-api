from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import io
import uuid
from typing import List, Dict
import aiofiles

app = FastAPI(title="PDF to Image Converter API")

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONVERT_API_SECRET = os.getenv("CONVERT_API_SECRET", "change_me")
TEMP_DIR = os.getenv("TEMP_DIR", "temp_files")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_images")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PDFConverter:
    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        self.base_url = "https://v2.convertapi.com/convert/pdf/to/jpg"

    async def convert_pdf_to_images(self, pdf_file: UploadFile) -> List[Dict]:
        temp_filename = f"{uuid.uuid4()}_{pdf_file.filename}"
        temp_path = os.path.join(TEMP_DIR, temp_filename)

        try:
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await pdf_file.read()
                await out_file.write(content)

            with open(temp_path, 'rb') as file:
                files = {'File': (pdf_file.filename, file, 'application/pdf')}
                params = {'Secret': self.api_secret, 'StoreFile': 'true'}
                response = requests.post(self.base_url, params=params, files=files)

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"Conversion failed: {response.text}")

                result = response.json()
                images_data = []

                for i, file_info in enumerate(result.get('Files', [])):
                    image_url = file_info['Url']
                    image_name = file_info['FileName']
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        images_data.append({
                            'image_id': i + 1,
                            'image_name': image_name,
                            'image_data': img_response.content,
                            'image_size': len(img_response.content),
                            'content_type': 'image/jpeg'
                        })
                return images_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

converter = PDFConverter(CONVERT_API_SECRET)

@app.get("/")
async def root():
    return {"message": "PDF to Image Converter API", "status": "running"}

@app.post("/convert-pdf")
async def convert_pdf_to_images(pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        images_data = await converter.convert_pdf_to_images(pdf_file)

        if not images_data:
            raise HTTPException(status_code=500, detail="No images were generated from the PDF")

        response_data = {
            "status": "success",
            "message": f"Successfully converted PDF to {len(images_data)} images",
            "total_images": len(images_data),
            "images": []
        }

        for img_data in images_data:
            image_id = img_data['image_id']
            image_filename = f"{uuid.uuid4()}_{img_data['image_name']}"
            image_path = os.path.join(OUTPUT_DIR, image_filename)

            async with aiofiles.open(image_path, 'wb') as f:
                await f.write(img_data['image_data'])

            response_data["images"].append({
                "image_id": image_id,
                "image_name": img_data['image_name'],
                "download_url": f"/download-image/{image_filename}",
                "preview_url": f"/preview-image/{image_filename}",
                "size_kb": round(img_data['image_size'] / 1024, 2)
            })

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download-image/{image_filename}")
async def download_image(image_filename: str):
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    async def iterfile():
        async with aiofiles.open(image_path, 'rb') as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(iterfile(), media_type='image/jpeg')

@app.get("/preview-image/{image_filename}")
async def preview_image(image_filename: str):
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    async def iterfile():
        async with aiofiles.open(image_path, 'rb') as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(iterfile(), media_type='image/jpeg')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
