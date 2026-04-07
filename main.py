import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Career & Education Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models & data ────────────────────────────────────────────────────────
stream_model  = joblib.load("model/stream_model.pkl")
stream_scaler = joblib.load("model/stream_scaler.pkl")
college_data  = pd.read_csv("colleges_data.csv")

# ── Schema ────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    quantitative_score:         float = Field(..., ge=0, le=100)
    logical_score:              float = Field(..., ge=0, le=100)
    verbal_score:               float = Field(..., ge=0, le=100)
    creative_score:             float = Field(..., ge=0, le=100)
    technical_score:            float = Field(..., ge=0, le=100)
    aggregate_percentage:       float = Field(..., ge=0, le=100)
    state:                      str
    prefers_government_college: bool
    top_n:                      Optional[int] = Field(5, ge=1, le=20)

# ── Helpers ───────────────────────────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    # 1. Predict stream
    aptitude = [[
        payload.quantitative_score,
        payload.logical_score,
        payload.verbal_score,
        payload.creative_score,
        payload.technical_score,
    ]]
    scaled           = stream_scaler.transform(aptitude)
    predicted_stream = stream_model.predict(scaled)[0]

    # 2. Filter colleges by stream
    candidates = college_data[
        college_data["stream_offered"].str.contains(
            predicted_stream, case=False, na=False, regex=False
        )
    ].copy()

    if candidates.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No colleges found for stream: {predicted_stream}"
        )

    # 3. Score each college
    results = []
    for _, college in candidates.iterrows():
        diff         = payload.aggregate_percentage - college["last_year_cutoff"]
        academic_fit = clamp((diff + 20) / 40, 0.0, 1.0)

        location_fit = 1.0 if payload.state.strip().lower() == str(college["college_state"]).strip().lower() else 0.5

        college_type = str(college.get("college_type", ""))
        if payload.prefers_government_college:
            pref_fit = 1.0 if "government" in college_type.lower() else 0.0
        else:
            pref_fit = 1.0 if "private" in college_type.lower() else 0.0

        final_score = (
            0.60 * academic_fit +
            0.25 * location_fit +
            0.15 * pref_fit
        )

        results.append({
            "college_name":      college["college_name"],
            "college_state":     college["college_state"],
            "college_type":      college_type,
            "last_year_cutoff":  college["last_year_cutoff"],
            "suitability_score": round(final_score, 4),
        })

    # 4. Sort and return top N
    results.sort(key=lambda x: x["suitability_score"], reverse=True)
    top = results[:payload.top_n]

    return {
        "predicted_stream":          predicted_stream,
        "total_colleges_evaluated":  len(results),
        "recommendations": [
            {"rank": i + 1, **r} for i, r in enumerate(top)
        ]
    }
