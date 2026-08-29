import io
import json
import sqlite3
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from features import extract_features
from model import (
    load_autoencoder,
    load_fusion_model,
    reconstruction_error,
    predict_quality
)

app = FastAPI(title="Image Quality Assessment")
# SQLite database
db = sqlite3.connect(
    "image_quality.db",
    check_same_thread=False
)
db.execute("""
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    quality_score REAL,
    quality_label TEXT,
    issues TEXT,
    created_at TEXT
)
""")
db.commit()

# Load trained models
try:
    autoencoder, device = load_autoencoder()
    classifier = load_fusion_model()
    models_ready = True
except Exception as e:
    print("Model loading error:", e)
    models_ready = False

@app.get("/")
def home():
    return {
        "message": "Image Quality Assessment API"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image"
        )
    try:
        data = await file.read()

        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

        image = np.array(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the image"
        )
    if not models_ready:
        raise HTTPException(
            status_code=503,
            detail="Models are not available"
        )
    # Extract traditional image-quality features
    features = extract_features(image)
    # Get reconstruction error from autoencoder
    error = reconstruction_error(
        autoencoder,
        image,
        device
    )
    features["reconstruction_error"] = error
    # Final prediction from Random Forest
    label, probabilities = predict_quality(
        classifier,
        features
    )
    label = str(label)
    # Convert class probabilities into a 0-100 score
    score = (
        probabilities.get("ACCEPTABLE", 0) * 100
        + probabilities.get("DEGRADED", 0) * 55
        + probabilities.get("DEFECTIVE", 0) * 10
    )
    score = round(score, 2)
    issues = []

    # Blur
    if features["sharpness"] < 0.40:
        issues.append({
            "type": "blur",
            "severity": "high",
            "confidence": round(
                1 - features["sharpness"],
                2
            )
        })

    # Underexposure
    if features["underexposed_ratio"] > 0.20:
        issues.append({
            "type": "underexposure",
            "severity": "medium",
            "confidence": round(
                min(
                    features["underexposed_ratio"] * 2,
                    1
                ),
                2
            )
        })

    # Overexposure
    if features["overexposed_ratio"] > 0.20:
        issues.append({
            "type": "overexposure",
            "severity": "medium",
            "confidence": round(
                min(
                    features["overexposed_ratio"] * 2,
                    1
                ),
                2
            )
        })

    # Noise
    if features["noise_level"] > 10:
        issues.append({
            "type": "noise",
            "severity": "medium",
            "confidence": round(
                min(
                    features["noise_level"] / 20,
                    1
                ),
                2
            )
        })

    # Severe degradation / possible defect
    if error > 0.05:
        issues.append({
            "type": "visual_defect",
            "severity": "high",
            "confidence": round(
                min(error * 10, 1),
                2
            )
        })

    # Save the analysis
    db.execute(
        """
        INSERT INTO analyses
        (filename, quality_score, quality_label,
         issues, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            file.filename,
            score,
            label,
            json.dumps(issues),
            datetime.now().isoformat()
        )
    )
    db.commit()
    # Keep API response simple as required
    return {
        "quality_score": score,
        "quality_label": label,
        "issues": issues
    }

@app.get("/history")
def get_history():
    rows = db.execute("""
        SELECT id, filename, quality_score,
               quality_label, issues, created_at
        FROM analyses
        ORDER BY id DESC
    """).fetchall()
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "filename": row[1],
            "quality_score": row[2],
            "quality_label": row[3],
            "issues": json.loads(row[4]),
            "created_at": row[5]
        })
    return history