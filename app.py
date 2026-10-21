"""
app.py
------
FastAPI production backend for Handwritten Number Sequence Recognizer.
Replaces Streamlit with a high-performance, lightweight asynchronous API
and modern HTML5/Vanilla CSS/JS user interface.

Author: Naravula Prudhvi Sri Bhanu Vivek
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from utils import pil_from_data_url, pil_to_data_url, segment_and_preprocess_image

# ---------------------------------------------------------------------------
# Global model holder
# ---------------------------------------------------------------------------
model = None
MODEL_PATH = "digit_model.keras"


def get_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm model on startup
    get_model()
    yield


app = FastAPI(
    title="Handwritten Number Sequence Recognizer",
    description="Recognizes handwritten numbers & sequences in reading order using a CNN trained on MNIST.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class CanvasPayload(BaseModel):
    image: str  # Base64 data URL (e.g. data:image/png;base64,...)


def run_prediction_pipeline(pil_img: Image.Image) -> Dict:
    """
    Run segmentation, inference, and line grouping on any PIL image.
    """
    current_model = get_model()
    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure digit_model.keras is present."
        )

    start_time = time.time()
    segments = segment_and_preprocess_image(pil_img)

    if not segments:
        return {
            "success": False,
            "message": "No digits detected. Please draw or upload clearer numbers with good contrast.",
            "sequence": "",
            "total_digits": 0,
            "lines": [],
            "latency_ms": round((time.time() - start_time) * 1000, 1),
        }

    digits_info = []
    for item in segments:
        p1 = current_model.predict(item["tensor"], verbose=0)[0]
        p2 = current_model.predict(item.get("tensor_scaled", item["tensor"]), verbose=0)[0]
        # Multi-scale ensemble: gives optimal weight to standard crop + scaled crop
        probs = (p1 * 0.65 + p2 * 0.35)
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        preview_b64 = pil_to_data_url(item["preview_pil"])

        digits_info.append({
            "digit": digit,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 1),
            "probabilities": [round(float(p), 4) for p in probs],
            "preview_base64": preview_b64,
            "bbox": [int(v) for v in item["bbox"]],
            "line_idx": int(item.get("line_idx", 0)),
        })

    # Group by line index (Top-to-Bottom, Left-to-Right reading order)
    lines_dict: Dict[int, List[Dict]] = {}
    for d in digits_info:
        lines_dict.setdefault(d["line_idx"], []).append(d)

    lines_output = []
    line_strings = []
    for idx_order, (line_idx, line_digits) in enumerate(lines_dict.items()):
        line_str = "".join(str(d["digit"]) for d in line_digits)
        line_strings.append(line_str)
        lines_output.append({
            "line_number": idx_order + 1,
            "line_text": line_str,
            "digits": line_digits,
        })

    if len(line_strings) > 1:
        formatted_seq = "  |  ".join(f"Line {idx + 1}: {s}" for idx, s in enumerate(line_strings))
    else:
        formatted_seq = line_strings[0] if line_strings else ""

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "success": True,
        "sequence": formatted_seq,
        "raw_sequence": "".join(line_strings),
        "total_digits": len(digits_info),
        "lines": lines_output,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def get_index():
    index_path = os.path.join("templates", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend template index.html not found"})


@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": get_model() is not None,
        "framework": "FastAPI + Uvicorn",
    }


@app.post("/api/predict-canvas")
async def predict_canvas(payload: CanvasPayload):
    try:
        pil_img = pil_from_data_url(payload.image)
        return run_prediction_pipeline(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process canvas drawing: {str(e)}")


@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    try:
        pil_img = Image.open(file.file)
        return run_prediction_pipeline(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
