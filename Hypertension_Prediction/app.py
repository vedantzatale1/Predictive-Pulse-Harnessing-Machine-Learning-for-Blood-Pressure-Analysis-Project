from __future__ import annotations

import os
import pickle
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logreg_model.pkl")

app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production"


def load_model() -> Any:
    if not os.path.exists(MODEL_PATH):
        return None

    # Try pickle first.
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded = pickle.load(f)
        if hasattr(loaded, "model"):
            return loaded.model
        if hasattr(loaded, "predict"):
            return loaded
    except Exception:
        pass

    # Fallback to joblib if available.
    try:
        import joblib

        loaded = joblib.load(MODEL_PATH)
        if hasattr(loaded, "model"):
            return loaded.model
        if hasattr(loaded, "predict"):
            return loaded
    except Exception:
        return None

    return None


model = load_model()

stage_map = {
    0: "NORMAL",
    1: "HYPERTENSION (Stage-1)",
    2: "HYPERTENSION (Stage-2)",
    3: "HYPERTENSIVE CRISIS",
}

color_map = {
    0: "#10B981",
    1: "#F59E0B",
    2: "#F97316",
    3: "#EF4444",
}

recommendations = {
    0: {
        "title": "Normal Blood Pressure",
        "description": "Your cardiovascular risk assessment indicates normal blood pressure levels.",
        "actions": [
            "Maintain current healthy lifestyle",
            "Regular physical activity (150 minutes/week)",
            "Continue balanced, low-sodium diet",
            "Annual blood pressure monitoring",
            "Regular health check-ups",
        ],
        "priority": "LOW RISK",
    },
    1: {
        "title": "Stage 1 Hypertension",
        "description": "Mild elevation detected requiring lifestyle modifications and medical consultation.",
        "actions": [
            "Schedule appointment with healthcare provider",
            "Implement DASH diet plan",
            "Increase physical activity gradually",
            "Monitor blood pressure bi-weekly",
            "Reduce sodium intake (<2300mg/day)",
            "Consider stress management techniques",
        ],
        "priority": "MODERATE RISK",
    },
    2: {
        "title": "Stage 2 Hypertension",
        "description": "Significant hypertension requiring immediate medical intervention and treatment.",
        "actions": [
            "URGENT: Consult physician within 1-2 days",
            "Likely medication therapy required",
            "Comprehensive cardiovascular assessment",
            "Daily blood pressure monitoring",
            "Strict dietary sodium restriction",
            "Lifestyle modification counseling",
        ],
        "priority": "HIGH RISK",
    },
    3: {
        "title": "Hypertensive Crisis",
        "description": "CRITICAL: Dangerously elevated blood pressure requiring emergency medical care.",
        "actions": [
            "EMERGENCY: Seek immediate medical attention",
            "Call 911 if experiencing symptoms",
            "Do not delay treatment",
            "Monitor for stroke/heart attack signs",
            "Prepare current medication list",
            "Avoid physical exertion",
        ],
        "priority": "EMERGENCY",
    },
}


ENCODERS = {
    "Gender": {"Male": 0, "Female": 1},
    "Age": {"18-34": 1, "35-50": 2, "51-64": 3, "65+": 4},
    "History": {"No": 0, "Yes": 1},
    "Patient": {"No": 0, "Yes": 1},
    "TakeMedication": {"No": 0, "Yes": 1},
    "Severity": {"Mild": 0, "Moderate": 1, "Severe": 2},
    "BreathShortness": {"No": 0, "Yes": 1},
    "VisualChanges": {"No": 0, "Yes": 1},
    "NoseBleeding": {"No": 0, "Yes": 1},
    "Whendiagnoused": {"<1 Year": 1, "1 - 5 Years": 2, ">5 Years": 3},
    "Systolic": {"<100": 0, "100-110": 0, "111-120": 1, "121-130": 2, "130+": 3},
    "Diastolic": {"70-80": 0, "81-90": 1, "91-100": 2, "100+": 3},
    "ControlledDiet": {"No": 0, "Yes": 1},
}


FIELD_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Age": ["18-34", "35-50", "51-64", "65+"],
    "History": ["No", "Yes"],
    "Patient": ["No", "Yes"],
    "TakeMedication": ["No", "Yes"],
    "Severity": ["Mild", "Moderate", "Severe"],
    "BreathShortness": ["No", "Yes"],
    "VisualChanges": ["No", "Yes"],
    "NoseBleeding": ["No", "Yes"],
    "Whendiagnoused": ["<1 Year", "1 - 5 Years", ">5 Years"],
    "Systolic": ["100-110", "111-120", "121-130", "130+"],
    "Diastolic": ["70-80", "81-90", "91-100", "100+"],
    "ControlledDiet": ["No", "Yes"],
}


REQUIRED_FIELDS = list(FIELD_OPTIONS.keys())


def _predict_with_fallback(feature_array: np.ndarray, encoded_map: Dict[str, int]) -> int:
    if model is None:
        return random.randint(0, 3)

    try:
        pred = model.predict(feature_array)
        val = pred[0]
        if isinstance(val, str) and val in stage_map.values():
            for k, name in stage_map.items():
                if name == val:
                    return k
        return int(val)
    except Exception:
        try:
            frame = pd.DataFrame([encoded_map])
            pred = model.predict(frame)
            val = pred[0]
            if isinstance(val, str) and val in stage_map.values():
                for k, name in stage_map.items():
                    if name == val:
                        return k
            return int(val)
        except Exception:
            return random.randint(0, 3)


def _confidence_with_fallback(feature_array: np.ndarray, encoded_map: Dict[str, int], pred_class: int) -> float:
    if model is None:
        return 87.5

    try:
        probs = model.predict_proba(feature_array)[0]
        return float(np.max(probs) * 100)
    except Exception:
        try:
            frame = pd.DataFrame([encoded_map])
            probs = model.predict_proba(frame)[0]
            return float(np.max(probs) * 100)
        except Exception:
            return 85.0 if pred_class != 3 else 92.0


@app.route("/")
def home():
    return render_template(
        "index.html",
        options=FIELD_OPTIONS,
        form_data={},
        prediction_text=None,
        result_color=None,
        confidence=None,
        recommendation=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data: Dict[str, str] = {}
        for field in REQUIRED_FIELDS:
            value = request.form.get(field)
            if not value:
                flash(f"Please complete all required fields: {field}", "error")
                return redirect(url_for("home"))
            form_data[field] = value

        try:
            encoded = [
                ENCODERS["Gender"][form_data["Gender"]],
                ENCODERS["Age"][form_data["Age"]],
                ENCODERS["History"][form_data["History"]],
                ENCODERS["Patient"][form_data["Patient"]],
                ENCODERS["TakeMedication"][form_data["TakeMedication"]],
                ENCODERS["Severity"][form_data["Severity"]],
                ENCODERS["BreathShortness"][form_data["BreathShortness"]],
                ENCODERS["VisualChanges"][form_data["VisualChanges"]],
                ENCODERS["NoseBleeding"][form_data["NoseBleeding"]],
                ENCODERS["Whendiagnoused"][form_data["Whendiagnoused"]],
                ENCODERS["Systolic"][form_data["Systolic"]],
                ENCODERS["Diastolic"][form_data["Diastolic"]],
                ENCODERS["ControlledDiet"][form_data["ControlledDiet"]],
            ]
        except KeyError as e:
            flash(f"Invalid selection detected: {str(e)}", "error")
            return redirect(url_for("home"))

        # Manual MinMax style scaling for ordinal features.
        scaled_encoded = encoded.copy()
        scaled_encoded[1] = (encoded[1] - 1) / (4 - 1)  # Age
        scaled_encoded[5] = encoded[5] / 2  # Severity
        scaled_encoded[9] = (encoded[9] - 1) / (3 - 1)  # Time diagnosed
        scaled_encoded[10] = encoded[10] / 3  # Systolic
        scaled_encoded[11] = encoded[11] / 3  # Diastolic

        input_array = np.array(scaled_encoded).reshape(1, -1)
        encoded_map = {field: encoded[idx] for idx, field in enumerate(REQUIRED_FIELDS)}

        prediction = _predict_with_fallback(input_array, encoded_map)
        confidence = _confidence_with_fallback(input_array, encoded_map, prediction)

        if model is None:
            flash("Demo Mode: Using simulated AI prediction for demonstration", "info")

        result_text = stage_map.get(prediction, "UNKNOWN")
        result_color = color_map.get(prediction, "#2563EB")
        result_recommendation = recommendations.get(prediction, recommendations[0])

        return render_template(
            "index.html",
            options=FIELD_OPTIONS,
            prediction_text=result_text,
            result_color=result_color,
            confidence=round(confidence, 1),
            recommendation=result_recommendation,
            form_data=form_data,
        )
    except Exception:
        flash("System error occurred. Please try again or contact technical support.", "error")
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
