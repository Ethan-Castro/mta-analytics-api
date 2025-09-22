from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import traceback
import os

# Optional S3 fetch (kept for flexibility)
import boto3
from botocore.exceptions import BotoCoreError, ClientError

FEATURE_ORDER = [
    "month",
    "hour_of_day",
    "is_weekend",
    "is_rush_hour",
    "road_distance",
    "distance_to_nearest_cuny",
    "is_2025",
]

# ---------- helpers: local-first, with optional S3 fallback ----------
def all_artifacts_present_root() -> bool:
    """Check artifacts at repo root (same dir as app.py)."""
    needed = [
        "bus_speed_predictor.pkl",
        "speed_scaler.pkl",
        "violation_predictor.pkl",
        "model_registry.json",
        "preprocessing_params.json",
    ]
    return all(os.path.exists(n) for n in needed)

def ensure_models_local_via_s3_to_root():
    """If S3 env vars are set and local files are missing, download to repo root."""
    bucket = os.getenv("MODEL_S3_BUCKET")
    prefix = os.getenv("MODEL_S3_PREFIX")
    region = os.getenv("AWS_REGION", "us-east-1")
    if not bucket or not prefix:
        print("No MODEL_S3_BUCKET/MODEL_S3_PREFIX set; skipping S3 download.")
        return

    files = [
        "bus_speed_predictor.pkl",
        "speed_scaler.pkl",
        "violation_predictor.pkl",
        "model_registry.json",
        "preprocessing_params.json",
    ]
    s3 = boto3.client("s3", region_name=region)

    # Optional: quick visibility
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        sample = [o["Key"] for o in resp.get("Contents", [])][:10] if resp.get("Contents") else []
        print("DEBUG S3 list (sample):", sample)
    except Exception as e:
        print("DEBUG S3 list failed:", e)

    for fname in files:
        local = fname  # download into repo root
        if os.path.exists(local):
            continue
        key = f"{prefix.rstrip('/')}/{fname}"
        try:
            print(f"Downloading s3://{bucket}/{key} -> {local}")
            s3.download_file(bucket, key, local)
        except (BotoCoreError, ClientError) as e:
            print(f"⚠️ Failed to download {fname} from S3:", e)

def load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

# ---------- Flask app ----------
app = Flask(__name__, template_folder="templates", static_folder=None)
CORS(app)

# Startup: local-first, then optional S3
print("Boot: ensuring artifacts present …")
if not all_artifacts_present_root():
    ensure_models_local_via_s3_to_root()
else:
    print("Found all artifacts locally at repo root; skipping S3.")

print("Loading models …")
speed_model = None
speed_scaler = None
violation_model = None
preprocessing_params = {}
model_registry = {
    "name": "MTA Bus Analytics API",
    "version": "0.0.0",
    "created_at": datetime.now().isoformat(),
    "models": {},
}

try:
    if os.path.exists("bus_speed_predictor.pkl"):
        speed_model = joblib.load("bus_speed_predictor.pkl")
    if os.path.exists("speed_scaler.pkl"):
        speed_scaler = joblib.load("speed_scaler.pkl")
    if os.path.exists("violation_predictor.pkl"):
        violation_model = joblib.load("violation_predictor.pkl")
    preprocessing_params = load_json("preprocessing_params.json", default={"cuny_routes": []})
    model_registry = load_json("model_registry.json", default=model_registry)
    print("✅ Models loaded successfully!")
    print("Loaded flags:", {
        "speed_model": speed_model is not None,
        "speed_scaler": speed_scaler is not None,
        "violation_model": violation_model is not None,
        "has_params": bool(preprocessing_params),
    })
except Exception as e:
    print("⚠️ Warning: one or more artifacts failed to load:", e)

@app.route("/")
def home():
    return jsonify({
        "name": "MTA Bus Analytics API",
        "version": model_registry.get("version", "0.0.0"),
        "endpoints": {
            "/ui": "Simple HTML dashboard",
            "/health": "Health/availability check",
            "/predict/speed": "POST: Predict bus speed",
            "/predict/violations": "POST: Forecast daily violations",
            "/analyze/route": "POST: Analyze route",
            "/models/info": "GET: Model metadata"
        }
    })

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "speed_model": speed_model is not None,
        "speed_scaler": speed_scaler is not None,
        "violation_model": violation_model is not None,
        "has_params": bool(preprocessing_params),
        "version": model_registry.get("version", "0.0.0")
    })

@app.route("/predict/speed", methods=["POST"])
def predict_speed():
    try:
        if speed_model is None or speed_scaler is None:
            return jsonify({"success": False, "error": "Speed model or scaler not loaded"}), 503

        data = request.get_json(force=True) or {}
        month = int(data.get("month", datetime.now().month))
        hour = int(data.get("hour", 12))
        is_weekend = int(bool(data.get("is_weekend", 0)))
        is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18, 19])
        road_distance = float(data.get("road_distance", 1.0))
        distance_to_cuny = float(data.get("distance_to_cuny", data.get("distance_to_nearest_cuny", 0.5)))
        is_2025 = 1

        X = pd.DataFrame([{
            "month": month,
            "hour_of_day": hour,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            "road_distance": road_distance,
            "distance_to_nearest_cuny": distance_to_cuny,
            "is_2025": is_2025
        }])[FEATURE_ORDER]

        Xs = speed_scaler.transform(X)
        pred = float(speed_model.predict(Xs)[0])
        margin = 2.5

        return jsonify({
            "success": True,
            "prediction": {
                "speed_mph": round(pred, 1),
                "confidence_interval": [round(pred - margin, 1), round(pred + margin, 1)],
                "traffic_level": "smooth" if pred > 10 else ("moderate" if pred > 7 else "congested")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400

@app.route("/predict/violations", methods=["POST"])
def predict_violations():
    try:
        if violation_model is None:
            return jsonify({"success": False, "error": "Violation model not loaded"}), 503

        data = request.get_json(force=True) or {}
        route_id = data.get("route_id", "M15")
        days_ahead = int(data.get("days_ahead", 7))

        rng = np.random.default_rng(42)
        base = int(data.get("base_violations", rng.integers(10, 50)))

        preds = []
        for d in range(days_ahead):
            t = datetime.now() + timedelta(days=d)
            feats = pd.DataFrame([{
                "day_of_week": t.weekday(),
                "month": t.month,
                "day_of_month": t.day,
                "is_weekend": int(t.weekday() >= 5),
                "violations_lag_1": base + int(rng.integers(-5, 5)),
                "violations_lag_7": base + int(rng.integers(-10, 10)),
                "violations_lag_14": base + int(rng.integers(-10, 10))
            }])
            yhat = float(violation_model.predict(feats)[0])
            yint = max(0, int(round(yhat)))
            preds.append({
                "date": t.strftime("%Y-%m-%d"),
                "day_name": t.strftime("%A"),
                "predicted_violations": yint,
                "severity": "high" if yhat > 40 else ("medium" if yhat > 20 else "low")
            })

        avg_daily = float(np.mean([p["predicted_violations"] for p in preds])) if preds else 0.0
        return jsonify({
            "success": True,
            "route_id": route_id,
            "forecast": preds,
            "summary": {
                "total_expected": int(sum(p["predicted_violations"] for p in preds)),
                "peak_day": max(preds, key=lambda x: x["predicted_violations"])["date"] if preds else None,
                "average_daily": round(avg_daily, 1)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400

@app.route("/analyze/route", methods=["POST"])
def analyze_route():
    try:
        if speed_model is None or speed_scaler is None or violation_model is None:
            return jsonify({"success": False, "error": "Required model(s) not loaded"}), 503

        data = request.get_json(force=True) or {}
        route_id = data.get("route_id", "M15")

        now = datetime.now()
        hour = now.hour
        speed_df = pd.DataFrame([{
            "month": now.month,
            "hour_of_day": hour,
            "is_weekend": int(now.weekday() >= 5),
            "is_rush_hour": int(hour in [7, 8, 9, 16, 17, 18, 19]),
            "road_distance": float(data.get("road_distance", 1.5)),
            "distance_to_nearest_cuny": float(data.get("distance_to_cuny", 0.3)),
            "is_2025": 1
        }])[FEATURE_ORDER]
        current_speed = float(speed_model.predict(speed_scaler.transform(speed_df))[0])

        viol_df = pd.DataFrame([{
            "day_of_week": now.weekday(),
            "month": now.month,
            "day_of_month": now.day,
            "is_weekend": int(now.weekday() >= 5),
            "violations_lag_1": int(data.get("violations_lag_1", 25)),
            "violations_lag_7": int(data.get("violations_lag_7", 30)),
            "violations_lag_14": int(data.get("violations_lag_14", 28))
        }])
        expected_violations = float(violation_model.predict(viol_df)[0])
        expected_violations_int = max(0, int(round(expected_violations)))

        recs = []
        if current_speed < 7:
            recs.append({"priority": "high", "action": "Deploy additional enforcement",
                         "reason": f"Speed is {current_speed:.1f} mph, below acceptable threshold"})
        if expected_violations > 30:
            recs.append({"priority": "medium", "action": "Increase camera monitoring",
                         "reason": f"High violation count expected ({expected_violations_int} violations)"})

        return jsonify({
            "success": True,
            "analysis": {
                "route_id": route_id,
                "timestamp": now.isoformat(),
                "is_cuny_route": route_id in preprocessing_params.get("cuny_routes", []),
                "predictions": {
                    "current_speed": {
                        "value": round(current_speed, 1),
                        "unit": "mph",
                        "status": "good" if current_speed > 10 else ("moderate" if current_speed > 7 else "poor")
                    },
                    "expected_violations_today": {
                        "value": expected_violations_int,
                        "severity": "high" if expected_violations > 40 else ("medium" if expected_violations > 20 else "low")
                    }
                },
                "recommendations": recs
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400

@app.route("/models/info", methods=["GET"])
def get_models_info():
    return jsonify(model_registry)

if __name__ == "__main__":
    # Render uses gunicorn via Dockerfile CMD, but keep this for local dev
    app.run(debug=True, host="0.0.0.0", port=5000)
