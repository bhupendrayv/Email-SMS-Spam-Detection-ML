"""
Production FastAPI Backend for Email & SMS Spam Detection, Phishing, and URL Security.

Endpoints:
- POST /predict/sms
- POST /predict/email
- POST /predict/phishing
- POST /predict/url
- GET  /stats
- GET  /history
- GET  /models/compare
- POST /feedback

Standardized Prediction JSON response:
{
    "label": "Spam" | "Legitimate (Ham)",
    "spam_probability": float (0.0 to 1.0),
    "confidence": float (0.0 to 1.0),
    "top_keywords": [{"word": str, "score": float, "sentiment": str}],
    "model_used": str
}

Includes:
- Robust Pydantic request models with input validation
- Error handling with structured HTTP exceptions
- CORS middleware
- Centralized structured logging
- Models loaded once at startup
"""

from __future__ import annotations

import datetime
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("spam_detection_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.predict import SpamPredictor
from src.advanced_detectors import PhishingDetector, MaliciousURLDetector
from src.database import db

# Global in-memory state & loaded models
ml_resources: Dict[str, Any] = {}
prediction_history_store: List[Dict[str, Any]] = []
feedback_store: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load machine learning models once at startup.
    """
    logger.info("Initializing and caching threat detection models into memory...")
    try:
        predictor = SpamPredictor(model_name="svm_rbf", representation="tfidf")
        phishing_detector = PhishingDetector(spam_detector=predictor)
        url_detector = MaliciousURLDetector()

        ml_resources["predictor"] = predictor
        ml_resources["phishing_detector"] = phishing_detector
        ml_resources["url_detector"] = url_detector
        logger.info("All models loaded successfully into memory at startup.")
    except Exception as e:
        logger.error(f"Failed to load ML models at startup: {e}", exc_info=True)
        raise RuntimeError(f"Startup model initialization failed: {e}")

    yield

    logger.info("Shutting down threat detection API.")
    ml_resources.clear()


app = FastAPI(
    title="Email & SMS Spam & Cyber Threat Detection API",
    description="High-performance, production API for classifying SMS, Email, Phishing, and Malicious URLs.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Request & Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class SMSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Raw SMS message text")

    @field_validator("text")
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("SMS text cannot be empty or solely whitespace.")
        return v.strip()


class EmailRequest(BaseModel):
    subject: str = Field(default="", max_length=1000, description="Email subject line")
    body: str = Field(..., min_length=1, max_length=50000, description="Email message body")

    @field_validator("body")
    def validate_body(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Email body cannot be empty or solely whitespace.")
        return v.strip()


class PhishingRequest(BaseModel):
    subject: str = Field(default="", max_length=1000, description="Email subject line")
    body: str = Field(..., min_length=1, max_length=50000, description="Email message body")

    @field_validator("body")
    def validate_body(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Phishing body text cannot be empty or whitespace.")
        return v.strip()


class URLRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048, description="URL to be scanned")

    @field_validator("url")
    def validate_url(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("URL cannot be empty.")
        return v.strip()


class FeedbackRequest(BaseModel):
    message_content: str = Field(..., min_length=1, max_length=10000, description="Original content evaluated")
    predicted_label: str = Field(..., description="'Spam' or 'Legitimate (Ham)'")
    actual_label: str = Field(..., description="'Spam' or 'Legitimate (Ham)'")
    comments: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("actual_label")
    def validate_actual(cls, v: str) -> str:
        valid = {"spam", "legitimate (ham)", "ham"}
        if v.lower() not in valid:
            raise ValueError(f"actual_label must be one of: {valid}")
        return "Spam" if v.lower() == "spam" else "Legitimate (Ham)"


class KeywordAttribution(BaseModel):
    word: str
    score: float
    sentiment: str


class StandardPredictionResponse(BaseModel):
    label: str
    spam_probability: float
    confidence: float
    top_keywords: List[KeywordAttribution]
    model_used: str
    details: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper Function to Record History
# ─────────────────────────────────────────────────────────────────────────────

def record_history_entry(channel: str, content: str, label: str, spam_prob: float, conf: float, model_used: str = "TFIDF + svm_rbf") -> int:
    rec_id = db.save_prediction(
        message_text=content,
        msg_type=channel,
        prediction=label,
        probability=round(spam_prob, 4),
        model_used=model_used
    )
    entry = {
        "id": rec_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "channel": channel,
        "content_snippet": content[:80] + "..." if len(content) > 80 else content,
        "label": label,
        "spam_probability": round(spam_prob, 4),
        "confidence": round(conf, 4)
    }
    prediction_history_store.append(entry)
    return rec_id


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"])
def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "online",
        "service": "Email & SMS Spam Threat Detection API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.post(
    "/predict/sms",
    response_model=StandardPredictionResponse,
    tags=["Prediction"],
    summary="Predict SMS spam"
)
def predict_sms(request: SMSRequest):
    try:
        predictor: SpamPredictor = ml_resources["predictor"]
        res = predictor.predict(request.text)

        spam_prob = float(res["spam_probability_pct"] / 100.0)
        confidence = float(res["confidence_pct"] / 100.0)

        record_history_entry(
            channel="SMS",
            content=request.text,
            label=res["prediction"],
            spam_prob=spam_prob,
            conf=confidence
        )

        return StandardPredictionResponse(
            label=res["prediction"],
            spam_probability=round(spam_prob, 4),
            confidence=round(confidence, 4),
            top_keywords=res["top_influencing_words"],
            model_used=f"{res['representation']} + {res['model_name']}",
            details={"cleaned_text": res["cleaned_text"]}
        )
    except Exception as e:
        logger.error(f"Error in /predict/sms: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/predict/email",
    response_model=StandardPredictionResponse,
    tags=["Prediction"],
    summary="Predict Email spam (Subject + Body)"
)
def predict_email(request: EmailRequest):
    try:
        predictor: SpamPredictor = ml_resources["predictor"]
        res = predictor.predict_email(request.subject, request.body)

        spam_prob = float(res["spam_probability_pct"] / 100.0)
        confidence = float(res["confidence_pct"] / 100.0)

        record_history_entry(
            channel="Email",
            content=f"{request.subject} | {request.body}",
            label=res["prediction"],
            spam_prob=spam_prob,
            conf=confidence
        )

        return StandardPredictionResponse(
            label=res["prediction"],
            spam_probability=round(spam_prob, 4),
            confidence=round(confidence, 4),
            top_keywords=res["top_influencing_words"],
            model_used=f"{res['representation']} + {res['model_name']}",
            details={"cleaned_text": res["cleaned_text"]}
        )
    except Exception as e:
        logger.error(f"Error in /predict/email: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/predict/phishing",
    response_model=StandardPredictionResponse,
    tags=["Prediction"],
    summary="Predict phishing email threat"
)
def predict_phishing(request: PhishingRequest):
    try:
        phish_det: PhishingDetector = ml_resources["phishing_detector"]
        res = phish_det.predict(request.subject, request.body)

        phish_prob = float(res["phishing_probability"] / 100.0)
        label = "Spam" if res["is_phishing"] else "Legitimate (Ham)"
        conf = phish_prob if res["is_phishing"] else (1.0 - phish_prob)

        # Build keywords from indicators
        top_kws = []
        for kw in res["indicators"]["urgency_triggers"]:
            top_kws.append({"word": kw, "score": 2.5, "sentiment": "Urgency Trigger"})
        for kw in res["indicators"]["credential_triggers"]:
            top_kws.append({"word": kw, "score": 3.0, "sentiment": "Credential Request"})

        record_history_entry(
            channel="Phishing",
            content=f"{request.subject} | {request.body}",
            label=label,
            spam_prob=phish_prob,
            conf=conf
        )

        return StandardPredictionResponse(
            label=label,
            spam_probability=round(phish_prob, 4),
            confidence=round(conf, 4),
            top_keywords=top_kws[:10],
            model_used="Hybrid NLP + Threat Heuristics",
            details={
                "verdict": res["verdict"],
                "risk_score": res["threat_risk_score"],
                "indicators": res["indicators"]
            }
        )
    except Exception as e:
        logger.error(f"Error in /predict/phishing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/predict/url",
    response_model=StandardPredictionResponse,
    tags=["Prediction"],
    summary="Predict malicious or phishing URL"
)
def predict_url(request: URLRequest):
    try:
        url_det: MaliciousURLDetector = ml_resources["url_detector"]
        res = url_det.predict(request.url)

        threat_prob = float(res["threat_probability"] / 100.0)
        label = "Spam" if res["is_malicious"] else "Legitimate (Ham)"
        conf = threat_prob if res["is_malicious"] else (1.0 - threat_prob)

        top_kws = []
        for kw in res["lexical_features"]["suspicious_keywords"]:
            top_kws.append({"word": kw, "score": 2.0, "sentiment": "Suspicious URL Token"})

        record_history_entry(
            channel="URL",
            content=request.url,
            label=label,
            spam_prob=threat_prob,
            conf=conf
        )

        return StandardPredictionResponse(
            label=label,
            spam_probability=round(threat_prob, 4),
            confidence=round(conf, 4),
            top_keywords=top_kws,
            model_used="Char n-gram TF-IDF + Lexical Heuristics",
            details={
                "verdict": res["verdict"],
                "lexical_features": res["lexical_features"]
            }
        )
    except Exception as e:
        logger.error(f"Error in /predict/url: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/stats", tags=["Analytics"], summary="Get aggregated inspection statistics")
def get_stats():
    db_stats = db.get_stats()
    return {
        "total_checked": db_stats["total_checked"],
        "spam_count": db_stats["spam_count"],
        "ham_count": db_stats["ham_count"],
        "spam_percentage": db_stats["spam_percentage"],
        "inspections_by_channel": db_stats["breakdown_by_type"],
        "feedback_received_count": db_stats["user_feedback_count"],
        "db_backend": db_stats["db_backend"]
    }


@app.get("/history", tags=["Analytics"], summary="Get prediction history log")
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    db_history = db.get_history(limit=limit)
    return {
        "total_records": len(db_history),
        "history": db_history
    }


@app.get("/models/compare", tags=["Benchmarking"], summary="Get BoW vs TF-IDF comparison metrics")
def get_model_comparison():
    eval_csv_path = REPORTS_DIR / "evaluation_comparison_table.csv"
    if not eval_csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model comparison data not found. Please run evaluation benchmark first."
        )

    df = pd.read_csv(eval_csv_path)
    records = df.to_dict(orient="records")

    # Group by representation
    bow_models = [r for r in records if r["representation"] == "BOW"]
    tfidf_models = [r for r in records if r["representation"] == "TFIDF"]

    return {
        "total_configurations_tested": len(records),
        "best_model_overall": sorted(records, key=lambda x: x["f1_score"], reverse=True)[0],
        "all_models": records,
        "bow_leaderboard": sorted(bow_models, key=lambda x: x["f1_score"], reverse=True),
        "tfidf_leaderboard": sorted(tfidf_models, key=lambda x: x["f1_score"], reverse=True)
    }


@app.post("/feedback", tags=["Feedback"], summary="Submit user feedback on prediction")
def submit_feedback(feedback: FeedbackRequest):
    entry = {
        "id": len(feedback_store) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message_snippet": feedback.message_content[:100],
        "predicted_label": feedback.predicted_label,
        "actual_label": feedback.actual_label,
        "comments": feedback.comments,
        "is_correct": feedback.predicted_label.lower() == feedback.actual_label.lower()
    }
    feedback_store.append(entry)

    # Also record feedback in database
    db.save_prediction(
        message_text=feedback.message_content,
        msg_type="Feedback",
        prediction=feedback.predicted_label,
        probability=1.0 if feedback.predicted_label == "Spam" else 0.0,
        model_used="User Feedback Verification",
        user_feedback=f"Actual: {feedback.actual_label} | Notes: {feedback.comments or 'None'}"
    )

    logger.info(f"Feedback recorded [ID: {entry['id']}]: Correct={entry['is_correct']}")

    return {
        "status": "success",
        "message": "Feedback recorded successfully in persistent database!",
        "feedback_id": entry["id"]
    }

    return {
        "status": "success",
        "message": "Feedback recorded successfully. Thank you for helping improve model accuracy!",
        "feedback_id": entry["id"]
    }
