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

def adjust_stream(scores, model_pred):
    q, l, v, c, t = scores
    low, high = 30, 65
    weak_count = sum([q < low, l < low, v < low, c < low, t < low])

    if weak_count >= 3:
        return "Arts"
    if t >= high and q >= high:
        return "Science (PCM)"
    if c >= high and v >= high and t < high:
        return "Science (PCB)"
    if v >= high and c >= high:
        return "Humanities"
    if all(s >= 50 for s in [q, l, v, c, t]):
        return "Science (PCM)"
    if c >= 60 or t >= 60:
        return "Vocational"
    if q < high and l < high and v >= high and c < high and t < high:
        return "Commerce"
    return model_pred

def categorize(diff):
    if diff >= 10:
        return "Safe"
    elif diff >= 0:
        return "Target"
    elif diff >= -15:
        return "Reach"
    else:
        return "Reach"

def compute_academic_fit(diff):
    return max(0.0, min(1.0, (diff + 10) / 20))

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
    model_pred       = stream_model.predict(scaled)[0]

    predicted_stream = adjust_stream(
        [
            payload.quantitative_score,
            payload.logical_score,
            payload.verbal_score,
            payload.creative_score,
            payload.technical_score,
        ],
        model_pred
    )

    # 2. Filter colleges by stream
    stream_map = {
        "Science (PCM)": ["Science (PCM)", "Engineering"],
        "Science (PCB)": ["Science (PCB)"],
        "Humanities": ["Humanities", "Commerce"],
        "Arts": ["Arts", "Humanities"],
        "Commerce": ["Commerce", "Humanities"],
        "Vocational": ["Vocational"]
    }
    candidates = college_data[
        college_data["stream_offered"].isin(stream_map.get(predicted_stream, [predicted_stream]))
    ].copy()

    if candidates.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No realistic colleges found based on your score and cutoff"
        )

    # 3. Score each college
    safe, target, reach = [], [], []
    for _, college in candidates.iterrows():
        diff         = payload.aggregate_percentage - college["last_year_cutoff"]
        category     = categorize(diff)
        academic_fit = compute_academic_fit(diff)

        location_fit = 1.0 if payload.state.strip().lower() == str(college["college_state"]).strip().lower() else 0.5
        college_type = str(college.get("college_type", ""))
        pref_fit     = 1.0 if (payload.prefers_government_college and "government" in college_type.lower()) \
                          or (not payload.prefers_government_college and "private" in college_type.lower()) else 0.0

        final_score = round(0.7 * academic_fit + 0.2 * location_fit + 0.1 * pref_fit, 4)

        item = {
            "college_name":      college["college_name"],
            "college_state":     college["college_state"],
            "college_type":      college_type,
            "last_year_cutoff":  college["last_year_cutoff"],
            "score_difference":  round(diff, 2),
            "suitability_score": final_score
        }

        if category == "Safe":
            safe.append(item)
        elif category == "Target":
            target.append(item)
        else:
            reach.append(item)

    # Combine all internally
    combined = safe + target + reach
    combined.sort(key=lambda x: x["suitability_score"], reverse=True)
    top = combined[:payload.top_n]

    return {
        "predicted_stream": predicted_stream,
        "total_colleges_evaluated": len(combined),
        "recommendations": [{"rank": i+1, **r} for i, r in enumerate(top)]
    }
