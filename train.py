import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── 1. Load Data ──────────────────────────────────────────────────────────────
college_data = pd.read_csv("colleges_data.csv")
student_data = pd.read_csv("students_with_college.csv")

# ── 2. Merge ──────────────────────────────────────────────────────────────────
merged_df = pd.merge(
    left=student_data,
    right=college_data,
    left_on="assigned_college_id",
    right_on="college_id",
    how="left"
)
merged_df.drop("college_id", axis=1, inplace=True)

# ── 3. Clean ──────────────────────────────────────────────────────────────────
merged_df["device_access"].fillna("Unknown", inplace=True)
merged_df["extracurricular"].fillna("None", inplace=True)
merged_df["scholarships_available"].fillna("None", inplace=True)
merged_df.drop(columns=["admission_year_predicted"], inplace=True, errors="ignore")

critical_cols = [
    "quantitative_score", "logical_score", "verbal_score", "creative_score",
    "technical_score", "aggregate_percentage", "last_year_cutoff", "dropout_risk_score"
]
for col in critical_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

merged_df.dropna(subset=critical_cols, inplace=True)
merged_df.drop_duplicates(inplace=True)

# ── 4. Feature Engineering ────────────────────────────────────────────────────
merged_df["score_cutoff_difference"] = (
    merged_df["aggregate_percentage"] - merged_df["last_year_cutoff"]
)

# ── 5. Stream Recommendation Model ───────────────────────────────────────────
print("\n--- Training Stream Recommendation Model ---")

X_stream = merged_df[[
    "quantitative_score", "logical_score", "verbal_score",
    "creative_score", "technical_score"
]]
y_stream = merged_df["suggested_stream"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_stream, y_stream, test_size=0.20, random_state=42, stratify=y_stream
)

stream_scaler = StandardScaler()
X_train_s_scaled = stream_scaler.fit_transform(X_train_s)
X_test_s_scaled  = stream_scaler.transform(X_test_s)

# Compute class weights to fix imbalance (Mixed, Science PCB underrepresented)
classes = np.unique(y_train_s)
weights = compute_class_weight("balanced", classes=classes, y=y_train_s)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)

stream_model = LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    class_weight=class_weight_dict,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
)
stream_model.fit(X_train_s_scaled, y_train_s)

y_pred_s = stream_model.predict(X_test_s_scaled)
print(f"\nStream Model Accuracy: {accuracy_score(y_test_s, y_pred_s):.4f}")
print(classification_report(y_test_s, y_pred_s))

# ── 6. Save ───────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(stream_model,  "models/stream_model.pkl",  compress=3)
joblib.dump(stream_scaler, "models/stream_scaler.pkl", compress=3)

print("\n✅ Models saved to /models/")
for f in os.listdir("models"):
    size = os.path.getsize(f"models/{f}") / (1024 * 1024)
    print(f"  {f}: {size:.2f} MB")