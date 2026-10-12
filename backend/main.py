"""
FastAPI backend for helmet detection service
Integrates with YOLOv8 helmet detector
"""

import os
import io
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys

# Add parent directory to path to import detectors
sys.path.append(str(Path(__file__).parent.parent))
from helmet_detector import HelmetDetector
from phone_detector import PhoneDetector
from fire_smoke_detector import FireSmokeDetector
from fall_detector import FallDetector
from restricted_area_detector import RestrictedAreaDetector

# Global detector instances
helmet_detector = None
phone_detector = None
fire_smoke_detector = None
fall_detector = None
restricted_area_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the detectors on startup"""
    global helmet_detector, phone_detector, fire_smoke_detector, fall_detector, restricted_area_detector
    try:
        # Initialize helmet detector
        helmet_model_path = Path(__file__).parent.parent / "yolov8n.pt"
        helmet_detector = HelmetDetector(str(helmet_model_path), conf=0.4)
        print("✅ Helmet detector initialized successfully")
        
        # Initialize phone detector
        phone_model_path = Path(__file__).parent.parent / "yolo11s.pt"
        phone_detector = PhoneDetector(str(phone_model_path), conf=0.4)
        print("✅ Phone detector initialized successfully")
        
        # Initialize fire/smoke detector
        fire_model_path = Path(__file__).parent.parent / "fire_yolov8_best.pt"
        fire_smoke_detector = FireSmokeDetector(str(fire_model_path), conf=0.25)
        print("✅ Fire/Smoke detector initialized successfully")
        
        # Initialize fall detector
        fall_model_path = Path(__file__).parent.parent / "fall_yolov11.pt"
        fall_detector = FallDetector(str(fall_model_path), conf=0.4)
        print("✅ Fall detector initialized successfully")
        
        # Initialize restricted area detector
        restricted_model_path = Path(__file__).parent.parent / "best.pt"
        restricted_area_detector = RestrictedAreaDetector(str(restricted_model_path), conf=0.4)
        print("✅ Restricted area detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detectors: {e}")
        helmet_detector = None
        phone_detector = None
        fire_smoke_detector = None
        fall_detector = None
        restricted_area_detector = None
    yield

app = FastAPI(title="Safety Detection API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8081", "http://localhost:8080","http://localhost:8000", "http://127.0.0.1:8081", "http://127.0.0.1:8080"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResult(BaseModel):
    label: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    success: bool
    detections: List[DetectionResult]
    helmet_count: int = 0
    no_helmet_count: int = 0
    phone_count: int = 0
    remote_count: int = 0
    fire_count: int = 0
    smoke_count: int = 0
    fall_count: int = 0
    standing_count: int = 0
    sitting_count: int = 0
    lying_down_count: int = 0
    image_width: int
    image_height: int
    message: str
    announcement: str = ""




@app.get("/")
async def root():
    """Health check endpoint"""
    print("[DEBUG] Root endpoint accessed")
    return {"status": "healthy", "helmet_detector_loaded": helmet_detector is not None, "phone_detector_loaded": phone_detector is not None, "fire_smoke_detector_loaded": fire_smoke_detector is not None, "fall_detector_loaded": fall_detector is not None, "restricted_area_detector_loaded": restricted_area_detector is not None}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    print("[DEBUG] Health endpoint accessed")
    return {
        "status": "healthy" if all([helmet_detector, phone_detector, fire_smoke_detector, fall_detector, restricted_area_detector]) else "unhealthy",
        "helmet_detector_loaded": helmet_detector is not None,
        "phone_detector_loaded": phone_detector is not None,
        "fire_smoke_detector_loaded": fire_smoke_detector is not None,
        "fall_detector_loaded": fall_detector is not None,
        "restricted_area_detector_loaded": restricted_area_detector is not None,
        "helmet_model_classes": list(helmet_detector.class_names.values()) if helmet_detector else [],
        "phone_model_classes": list(phone_detector.class_names.values()) if phone_detector else [],
        "fire_smoke_model_classes": list(fire_smoke_detector.class_names.values()) if fire_smoke_detector else [],
        "fall_model_classes": list(fall_detector.class_names.values()) if fall_detector else [],
        "restricted_area_model_classes": restricted_area_detector.get_available_classes() if restricted_area_detector else []
    }

def decode_image(image_data: bytes) -> np.ndarray:
    """Decode image bytes to OpenCV format"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.post("/detect", response_model=DetectionResponse)
async def detect_helmets(file: UploadFile = File(...)):
    """
    Detect helmets in uploaded image
    Returns detection results and annotated image
    """
    print(f"[DEBUG] Received detection request for file: {file.filename}")
    print(f"[DEBUG] Content type: {file.content_type}")
    print(f"[DEBUG] File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
    
    if not helmet_detector:
        print("[DEBUG] Helmet detector not initialized")
        raise HTTPException(status_code=503, detail="Helmet detector not initialized")
    
    if not file.content_type.startswith('image/'):
        print(f"[DEBUG] Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        image_data = await file.read()
        print(f"[DEBUG] Read {len(image_data)} bytes of image data")
        
        img = decode_image(image_data)
        print(f"[DEBUG] Decoded image shape: {img.shape}")
        
        # Run detection
        detections = helmet_detector.detect(img)
        print(f"[DEBUG] Found {len(detections)} detections")
        
        # Trigger TTS announcement if needed
        announcement_msg = helmet_detector._maybe_announce(img, detections)
        
        # Process results
        helmet_count = sum(1 for d in detections if d["label"] == "helmet")
        no_helmet_count = sum(1 for d in detections if d["label"] == "no_helmet")
        
        # Convert to response format
        detection_results = []
        for det in detections:
            detection_results.append(DetectionResult(
                label=det["label"],
                confidence=det["conf"],
                box=list(det["box"])
            ))
        
        # Annotate image
        annotated_img = helmet_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        # Determine status message
        if no_helmet_count == 0 and helmet_count > 0:
            message = "✅ All workers wearing helmets"
        elif no_helmet_count > 0:
            message = f"⚠️ {no_helmet_count} worker(s) without helmets"
        else:
            message = "🔍 No workers detected"
        
        print(f"[DEBUG] Detection complete: {helmet_count} helmets, {no_helmet_count} no_helmets")
        print(f"[DEBUG] Sending response with {len(detection_results)} detections")
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            helmet_count=helmet_count,
            no_helmet_count=no_helmet_count,
            image_width=img.shape[1],
            image_height=img.shape[0],
            message=message,
            announcement=announcement_msg or ""
        )
        
    except Exception as e:
        print(f"[DEBUG] Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-visual")
async def detect_helmets_visual(file: UploadFile = File(...)):
    """
    Detect helmets and return annotated image as base64
    """
    if not helmet_detector:
        raise HTTPException(status_code=503, detail="Helmet detector not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        img = decode_image(image_data)
        detections = helmet_detector.detect(img)
        annotated_img = helmet_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        return {
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections_count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-phones", response_model=DetectionResponse)
async def detect_phones(file: UploadFile = File(...)):
    """
    Detect cell phones in uploaded image
    Returns detection results and annotated image
    """
    print(f"[DEBUG] Received phone detection request for file: {file.filename}")
    print(f"[DEBUG] Content type: {file.content_type}")
    print(f"[DEBUG] File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
    
    if not phone_detector:
        print("[DEBUG] Phone detector not initialized")
        raise HTTPException(status_code=503, detail="Phone detector not initialized")
    
    if not file.content_type.startswith('image/'):
        print(f"[DEBUG] Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        image_data = await file.read()
        print(f"[DEBUG] Read {len(image_data)} bytes of image data")
        
        img = decode_image(image_data)
        print(f"[DEBUG] Decoded image shape: {img.shape}")
        
        # Run detection
        detections = phone_detector.detect(img)
        print(f"[DEBUG] Found {len(detections)} phone detections")
        
        # Process results
        phone_count = sum(1 for d in detections if d["label"] == "cell phone")
        remote_count = sum(1 for d in detections if d["label"] == "remote")
        
        # Convert to response format
        detection_results = []
        for det in detections:
            detection_results.append(DetectionResult(
                label=det["label"],
                confidence=det["conf"],
                box=list(det["box"])
            ))
        
        # Annotate image
        annotated_img = phone_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        # Determine status message
        if phone_count > 0:
            message = f"📱 {phone_count} phone(s) detected"
        elif remote_count > 0:
            message = f"📺 {remote_count} remote(s) detected"
        else:
            message = "🔍 No phones detected"
        
        print(f"[DEBUG] Phone detection complete: {phone_count} phones, {remote_count} remotes")
        print(f"[DEBUG] Sending response with {len(detection_results)} detections")
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            helmet_count=0,
            no_helmet_count=0,
            phone_count=phone_count,
            remote_count=remote_count,
            image_width=img.shape[1],
            image_height=img.shape[0],
            message=message
        )
        
    except Exception as e:
        print(f"[DEBUG] Phone detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Phone detection failed: {str(e)}")

@app.post("/detect-phones-visual")
async def detect_phones_visual(file: UploadFile = File(...)):
    """
    Detect phones and return annotated image as base64
    """
    if not phone_detector:
        raise HTTPException(status_code=503, detail="Phone detector not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        img = decode_image(image_data)
        detections = phone_detector.detect(img)
        annotated_img = phone_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        return {
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections_count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phone detection failed: {str(e)}")

@app.post("/detect-fire-smoke", response_model=DetectionResponse)
async def detect_fire_smoke(file: UploadFile = File(...)):
    """
    Detect fire and smoke in uploaded image
    Returns detection results and annotated image
    """
    print(f"[DEBUG] Received fire/smoke detection request for file: {file.filename}")
    print(f"[DEBUG] Content type: {file.content_type}")
    print(f"[DEBUG] File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
    
    if not fire_smoke_detector:
        print("[DEBUG] Fire/Smoke detector not initialized")
        raise HTTPException(status_code=503, detail="Fire/Smoke detector not initialized")
    
    if not file.content_type.startswith('image/'):
        print(f"[DEBUG] Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        image_data = await file.read()
        print(f"[DEBUG] Read {len(image_data)} bytes of image data")
        
        img = decode_image(image_data)
        print(f"[DEBUG] Decoded image shape: {img.shape}")
        
        # Run detection
        detections = fire_smoke_detector.detect(img)
        print(f"[DEBUG] Found {len(detections)} fire/smoke detections")
        
        # Process results
        fire_count = sum(1 for d in detections if d["label"] == "fire")
        smoke_count = sum(1 for d in detections if d["label"] == "smoke")
        
        # Convert to response format
        detection_results = []
        for det in detections:
            detection_results.append(DetectionResult(
                label=det["label"],
                confidence=det["conf"],
                box=list(det["box"])
            ))
        
        # Annotate image
        annotated_img = fire_smoke_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        # Determine status message
        if fire_count > 0:
            message = f"🔥 {fire_count} fire(s) detected"
        elif smoke_count > 0:
            message = f"💨 {smoke_count} smoke(s) detected"
        else:
            message = "🔍 No fire or smoke detected"
        
        print(f"[DEBUG] Fire/Smoke detection complete: {fire_count} fires, {smoke_count} smokes")
        print(f"[DEBUG] Sending response with {len(detection_results)} detections")
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            helmet_count=0,
            no_helmet_count=0,
            phone_count=0,
            remote_count=0,
            fire_count=fire_count,
            smoke_count=smoke_count,
            image_width=img.shape[1],
            image_height=img.shape[0],
            message=message
        )
        
    except Exception as e:
        print(f"[DEBUG] Fire/Smoke detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fire/Smoke detection failed: {str(e)}")

@app.post("/detect-fire-smoke-visual")
async def detect_fire_smoke_visual(file: UploadFile = File(...)):
    """
    Detect fire and smoke and return annotated image as base64
    """
    if not fire_smoke_detector:
        raise HTTPException(status_code=503, detail="Fire/Smoke detector not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        img = decode_image(image_data)
        detections = fire_smoke_detector.detect(img)
        annotated_img = fire_smoke_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        return {
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections_count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fire/Smoke detection failed: {str(e)}")

@app.post("/detect-falls", response_model=DetectionResponse)
async def detect_falls(file: UploadFile = File(...)):
    """
    Detect falls in uploaded image
    Returns detection results and annotated image
    """
    print(f"[DEBUG] Received fall detection request for file: {file.filename}")
    print(f"[DEBUG] Content type: {file.content_type}")
    print(f"[DEBUG] File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
    
    if not fall_detector:
        print("[DEBUG] Fall detector not initialized")
        raise HTTPException(status_code=503, detail="Fall detector not initialized")
    
    if not file.content_type.startswith('image/'):
        print(f"[DEBUG] Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        image_data = await file.read()
        print(f"[DEBUG] Read {len(image_data)} bytes of image data")
        
        img = decode_image(image_data)
        print(f"[DEBUG] Decoded image shape: {img.shape}")
        
        # Run detection
        detections = fall_detector.detect(img)
        print(f"[DEBUG] Found {len(detections)} fall detections")
        
        # Process results
        fall_count = sum(1 for d in detections if d["label"] == "fall")
        standing_count = sum(1 for d in detections if d["label"] == "standing")
        sitting_count = sum(1 for d in detections if d["label"] == "sitting")
        lying_down_count = sum(1 for d in detections if d["label"] == "lying_down")
        
        # Convert to response format
        detection_results = []
        for det in detections:
            detection_results.append(DetectionResult(
                label=det["label"],
                confidence=det["conf"],
                box=list(det["box"])
            ))
        
        # Annotate image
        annotated_img = fall_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        # Determine status message
        if fall_count > 0:
            message = f"🚨 {fall_count} fall(s) detected!"
        elif standing_count > 0:
            message = f"👥 {standing_count} person(s) standing"
        elif sitting_count > 0:
            message = f"🪑 {sitting_count} person(s) sitting"
        elif lying_down_count > 0:
            message = f"😴 {lying_down_count} person(s) lying down"
        else:
            message = "🔍 No persons detected"
        
        print(f"[DEBUG] Fall detection complete: {fall_count} falls, {standing_count} standing, {sitting_count} sitting, {lying_down_count} lying down")
        print(f"[DEBUG] Sending response with {len(detection_results)} detections")
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            helmet_count=0,
            no_helmet_count=0,
            phone_count=0,
            remote_count=0,
            fire_count=0,
            smoke_count=0,
            fall_count=fall_count,
            standing_count=standing_count,
            sitting_count=sitting_count,
            lying_down_count=lying_down_count,
            image_width=img.shape[1],
            image_height=img.shape[0],
            message=message
        )
        
    except Exception as e:
        print(f"[DEBUG] Fall detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fall detection failed: {str(e)}")

@app.post("/detect-falls-visual")
async def detect_falls_visual(file: UploadFile = File(...)):
    """
    Detect falls and return annotated image as base64
    """
    if not fall_detector:
        raise HTTPException(status_code=503, detail="Fall detector not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        img = decode_image(image_data)
        detections = fall_detector.detect(img)
        annotated_img = fall_detector.draw(img, detections)
        annotated_base64 = encode_image_to_base64(annotated_img)
        
        return {
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections_count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fall detection failed: {str(e)}")

# Restricted Area Detection Endpoints
class RestrictedAreaRequest(BaseModel):
    area_coords: List[List[int]]  # List of [x, y] coordinates for polygon area
    selected_classes: List[str] = []
    alert_classes: List[str] = []

class EllipticalAreaRequest(BaseModel):
    center: List[int]  # [x, y] center coordinates
    axes: List[int]    # [width, height] axes lengths
    selected_classes: List[str] = []
    alert_classes: List[str] = []

@app.post("/restricted-area/set-polygon")
async def set_restricted_area_polygon(request: RestrictedAreaRequest):
    """Set restricted area as polygon"""
    if not restricted_area_detector:
        raise HTTPException(status_code=503, detail="Restricted area detector not initialized")
    
    try:
        restricted_area_detector.set_restricted_area(request.area_coords)
        return {"success": True, "message": "Restricted area set successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set restricted area: {str(e)}")

@app.post("/restricted-area/set-ellipse")
async def set_restricted_area_ellipse(request: EllipticalAreaRequest):
    """Set restricted area as ellipse"""
    if not restricted_area_detector:
        raise HTTPException(status_code=503, detail="Restricted area detector not initialized")
    
    try:
        restricted_area_detector.set_elliptical_restricted_area(
            center=tuple(request.center), axes=tuple(request.axes)
        )
        return {"success": True, "message": "Elliptical restricted area set successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set restricted area: {str(e)}")

@app.post("/restricted-area/detect")
async def detect_restricted_area(file: UploadFile = File(...)):
    """Detect objects in restricted area"""
    if not restricted_area_detector:
        raise HTTPException(status_code=503, detail="Restricted area detector not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        img = decode_image(image_data)
        
        # Get detection parameters from query params
        selected_classes = None  # All classes
        alert_classes = []  # No alerts by default
        
        result = restricted_area_detector.detect(img, selected_classes, alert_classes)
        
        # Convert to response format
        detections = []
        for obj in result["detected_objects"]:
            detections.append(DetectionResult(
                label=obj["class"],
                confidence=obj["confidence"],
                box=obj["bbox"]
            ))
        
        message = ""
        if result["near_restricted"]:
            message = "⚠️ Object detected near restricted area!"
        elif result["alert_triggered"]:
            message = "🚨 Alert triggered! Object in restricted area!"
        else:
            message = f"🔍 {len(detections)} object(s) detected"
        
        return DetectionResponse(
            success=True,
            detections=detections,
            helmet_count=0,
            no_helmet_count=0,
            phone_count=0,
            remote_count=0,
            fire_count=0,
            smoke_count=0,
            fall_count=0,
            standing_count=0,
            sitting_count=0,
            lying_down_count=0,
            image_width=img.shape[1],
            image_height=img.shape[0],
            message=message
        )
        
    except Exception as e:
        print(f"[DEBUG] Restricted area detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restricted area detection failed: {str(e)}")

@app.post("/restricted-area/detect-visual")
async def detect_restricted_area_visual(file: UploadFile = File(...)):
    """Detect objects in restricted area and return annotated image"""
    if not restricted_area_detector:
        raise HTTPException(status_code=503, detail="Restricted area detector not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        img = decode_image(image_data)
        
        result = restricted_area_detector.detect(img)
        annotated_base64 = encode_image_to_base64(result["annotated_frame"])
        
        return {
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections": result["detected_objects"],
            "near_restricted": result["near_restricted"],
            "alert_triggered": result["alert_triggered"],
            "classes_detected": result["classes_detected"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restricted area detection failed: {str(e)}")

@app.get("/restricted-area/classes")
async def get_restricted_area_classes():
    """Get available classes for restricted area detection"""
    if not restricted_area_detector:
        raise HTTPException(status_code=503, detail="Restricted area detector not initialized")
    
    return {"classes": restricted_area_detector.get_available_classes()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
