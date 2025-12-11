from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np
import uuid
import os
import aiofiles
import tempfile
import logging
import shutil
from pdf2image import convert_from_path
import io
from typing import List, Dict, Optional

app = FastAPI(
    title="Smart NID/ID Card Processor API",
    description="Extracts photo, signature, and information from NID/ID card PDFs",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
TEMP_DIR = tempfile.mkdtemp(prefix="nid_processor_")
OUT_DIR = os.path.join(TEMP_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

def detect_signature_region(image_path: str) -> Optional[str]:
    """
    Detect and extract signature from the bottom portion of the image
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        
        # Crop bottom 30% of image (signature area)
        crop_height = int(height * 0.30)
        signature_area = img[height - crop_height:, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(signature_area, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area
        min_area = 50
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return None
        
        # Get bounding box for all valid contours
        x_coords = []
        y_coords = []
        
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
        
        padding = 15
        x_min = max(0, min(x_coords) - padding)
        y_min = max(0, min(y_coords) - padding)
        x_max = min(signature_area.shape[1], max(x_coords) + padding)
        y_max = min(signature_area.shape[0], max(y_coords) + padding)
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        # Extract signature
        signature = signature_area[y_min:y_max, x_min:x_max]
        
        if signature.size == 0:
            return None
        
        # Save signature
        signature_id = f"{uuid.uuid4()}_signature.jpg"
        signature_path = os.path.join(OUT_DIR, signature_id)
        cv2.imwrite(signature_path, signature)
        
        return signature_id
    
    except Exception as e:
        logger.error(f"Error detecting signature: {str(e)}")
        return None

def detect_photo_region(image_path: str) -> Optional[str]:
    """
    Detect and extract photo from NID/ID card
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        
        # For NID cards, photo is typically in top-right quadrant
        # Adjust these ratios based on your NID format
        
        # Try top-right 25% width, 30% height
        photo_width = int(width * 0.25)
        photo_height = int(height * 0.30)
        photo_x = width - photo_width - int(width * 0.05)  # 5% margin
        photo_y = int(height * 0.10)  # 10% from top
        
        # Extract photo region
        photo_region = img[photo_y:photo_y + photo_height, photo_x:photo_x + photo_width]
        
        if photo_region.size == 0:
            return None
        
        # Enhance photo quality
        # Convert to PIL for enhancement
        photo_pil = Image.fromarray(cv2.cvtColor(photo_region, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(photo_pil)
        photo_pil = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(photo_pil)
        photo_pil = enhancer.enhance(1.1)
        
        # Save photo
        photo_id = f"{uuid.uuid4()}_photo.jpg"
        photo_path = os.path.join(OUT_DIR, photo_id)
        photo_pil.save(photo_path, "JPEG", quality=95)
        
        return photo_id
    
    except Exception as e:
        logger.error(f"Error detecting photo: {str(e)}")
        return None

def extract_text_regions(image_path: str) -> Dict[str, str]:
    """
    Extract text regions from NID (placeholder for OCR integration)
    """
    # This is a placeholder - you can integrate with Tesseract OCR or other services
    return {
        "name": "Extracted Name",
        "nid_number": "Extracted NID Number",
        "date_of_birth": "Extracted DOB",
        "father_name": "Extracted Father's Name",
        "mother_name": "Extracted Mother's Name"
    }

def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Cleaned up temp directory: {TEMP_DIR}")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")

@app.post("/convert-pdf")
async def process_nid_pdf(pdf_file: UploadFile = File(...)):
    """
    Process NID/ID card PDF and extract components
    """
    # Validate file
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    file_id = str(uuid.uuid4())
    temp_pdf_path = os.path.join(TEMP_DIR, f"{file_id}.pdf")
    
    try:
        # Save uploaded PDF
        async with aiofiles.open(temp_pdf_path, "wb") as f:
            content = await pdf_file.read()
            # Check file size (max 10MB)
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(400, "File too large. Maximum size is 10MB")
            await f.write(content)
        
        logger.info(f"Processing PDF: {pdf_file.filename}")
        
        # Convert PDF to images
        pages = convert_from_path(temp_pdf_path, dpi=200)
        
        if not pages:
            raise HTTPException(500, "No pages found in PDF")
        
        results = []
        
        # Process each page
        for page_num, page in enumerate(pages, 1):
            # Save full page image
            full_image_id = f"{file_id}_page_{page_num}.jpg"
            full_image_path = os.path.join(OUT_DIR, full_image_id)
            page.save(full_image_path, "JPEG", quality=90)
            
            # Get file size
            file_size_kb = os.path.getsize(full_image_path) / 1024
            
            # Extract signature
            signature_id = detect_signature_region(full_image_path)
            
            # Extract photo
            photo_id = detect_photo_region(full_image_path)
            
            # Extract text information (placeholder)
            text_info = extract_text_regions(full_image_path)
            
            page_result = {
                "page_number": page_num,
                "full_image": {
                    "image_id": f"{file_id}_full_{page_num}",
                    "image_name": full_image_id,
                    "download_url": f"/download-image/{full_image_id}",
                    "preview_url": f"/preview-image/{full_image_id}",
                    "size_kb": round(file_size_kb, 2)
                },
                "extracted_components": {
                    "signature": {
                        "available": signature_id is not None,
                        "image_id": signature_id,
                        "download_url": f"/download-image/{signature_id}" if signature_id else None,
                        "preview_url": f"/preview-image/{signature_id}" if signature_id else None
                    } if signature_id else None,
                    "photo": {
                        "available": photo_id is not None,
                        "image_id": photo_id,
                        "download_url": f"/download-image/{photo_id}" if photo_id else None,
                        "preview_url": f"/preview-image/{photo_id}" if photo_id else None
                    } if photo_id else None,
                    "text_info": text_info
                }
            }
            
            results.append(page_result)
        
        # Clean up PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        
        response = {
            "status": "success",
            "message": f"Successfully processed PDF with {len(pages)} pages",
            "total_pages": len(pages),
            "original_filename": pdf_file.filename,
            "processing_id": file_id,
            "pages": results,
            "extraction_summary": {
                "signatures_found": sum(1 for page in results if page["extracted_components"]["signature"]),
                "photos_found": sum(1 for page in results if page["extracted_components"]["photo"]),
                "total_images": len(pages)
            }
        }
        
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        # Clean up on error
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")

@app.get("/preview-image/{filename}")
async def preview_image(filename: str):
    """Preview an extracted image"""
    image_path = os.path.join(OUT_DIR, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(404, "Image not found")
    
    return FileResponse(
        image_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.get("/download-image/{filename}")
async def download_image(filename: str):
    """Download an extracted image"""
    image_path = os.path.join(OUT_DIR, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(404, "Image not found")
    
    return FileResponse(
        image_path,
        media_type="application/octet-stream",
        filename=filename
    )

@app.get("/extraction-results/{processing_id}")
async def get_extraction_results(processing_id: str):
    """Get results for a specific processing ID"""
    # This would retrieve cached results in a real implementation
    # For now, returns a placeholder
    return {
        "processing_id": processing_id,
        "status": "completed",
        "message": "Use the original response for download links"
    }

@app.post("/enhance-image")
async def enhance_image(image_file: UploadFile = File(...)):
    """Enhance an extracted image (signature/photo)"""
    try:
        # Read image
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(400, "Invalid image file")
        
        # Convert to PIL for enhancement
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Apply enhancements
        # 1. Contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        # 2. Sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # 3. Brightness
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Convert back to bytes
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create response
        enhanced_id = f"{uuid.uuid4()}_enhanced.jpg"
        enhanced_path = os.path.join(OUT_DIR, enhanced_id)
        
        with open(enhanced_path, "wb") as f:
            f.write(img_byte_arr)
        
        return {
            "status": "success",
            "message": "Image enhanced successfully",
            "enhanced_image": {
                "image_id": enhanced_id,
                "download_url": f"/download-image/{enhanced_id}",
                "preview_url": f"/preview-image/{enhanced_id}"
            }
        }
    
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise HTTPException(500, f"Failed to enhance image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smart NID/ID Card Processor",
        "version": "2.0.0",
        "temp_directory": TEMP_DIR,
        "output_directory": OUT_DIR
    }

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Smart NID/ID Card Processor API",
        "version": "2.0.0",
        "endpoints": {
            "process_pdf": {
                "method": "POST",
                "path": "/convert-pdf",
                "description": "Process NID/ID card PDF and extract photo, signature, and information"
            },
            "preview_image": {
                "method": "GET",
                "path": "/preview-image/{filename}",
                "description": "Preview an extracted image"
            },
            "download_image": {
                "method": "GET",
                "path": "/download-image/{filename}",
                "description": "Download an extracted image"
            },
            "enhance_image": {
                "method": "POST",
                "path": "/enhance-image",
                "description": "Enhance signature or photo quality"
            },
            "health_check": {
                "method": "GET",
                "path": "/health",
                "description": "Service health status"
            }
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    cleanup_temp_files()

# Startup logging
@app.on_event("startup")
async def startup_event():
    logger.info("Smart NID/ID Card Processor API starting up...")
    logger.info(f"Temporary directory: {TEMP_DIR}")
    logger.info(f"Output directory: {OUT_DIR}")
