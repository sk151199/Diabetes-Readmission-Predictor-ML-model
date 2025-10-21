# app.py — Streamlit GUI (Manual Entry Only) with Age Bracket support
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Diabetes Readmission Predictor", layout="wide")
st.title("🏥 Diabetes Readmission Predictor — Manual Entry")

# Age bracket choices (match dataset format exactly)
AGE_CHOICES = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

def bracket_to_mid(br):
    if not isinstance(br, str) or "-" not in br:
        return np.nan
    s = br.strip("[]()")
    lo, hi = s.split("-")
    try:
        return (float(lo) + float(hi)) / 2
    except:
        return np.nan

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

MODEL_PATH = "readmission_multiclass.pkl"  # change to your .pkl (binary or multiclass)
try:
    model = load_model(MODEL_PATH)
    st.success(f"✅ Loaded model: {MODEL_PATH}")
    st.caption(f"Classes: {getattr(model, 'classes_', 'N/A')}")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# =========================
# Inspect expected columns
# =========================
def get_expected_columns(model) -> Tuple[List[str], List[str]]:
    pre = model.named_steps.get("pre")
    if pre is None:
        raise ValueError("Model does not include a 'pre' step. Save the full Pipeline.")
    # transformers_: [('num', <pipeline>, num_cols), ('cat', <pipeline>, cat_cols)]
    num_cols = list(pre.transformers_[0][2]) if pre.transformers_ else []
    cat_cols = []
    for name, _, cols in pre.transformers_:
        if name == "cat":
            cat_cols = list(cols)
            break
    return num_cols, cat_cols

try:
    num_cols, cat_cols = get_expected_columns(model)
    expected_cols = num_cols + cat_cols
except Exception as e:
    st.error(f"Could not read expected columns: {e}")
    st.stop()

with st.expander("Expected feature columns", expanded=False):
    st.write(expected_cols)

def align_to_expected(df: pd.DataFrame, exp_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in exp_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[exp_cols]

def field_exists(name: str) -> bool:
    return name in expected_cols

# =========================
# Manual Input Form
# =========================
st.subheader("🧍 Enter Patient Details")

with st.form("manual_form"):
    c1, c2, c3 = st.columns(3)
    row = {}

    # --- Age inputs ---
    # If the model expects the raw bracket 'age', render the bracket picker
    if field_exists("age"):
        row["age"] = c1.selectbox("Age (bracket)", AGE_CHOICES, index=5)  # default [50-60)
    # If the model expects numeric 'age_mid', render the slider
    if field_exists("age_mid"):
        # If we also have 'age' (above), we will auto-fill age_mid from the bracket after submit
        row["age_mid"] = c1.slider("Age midpoint", 0, 100, 55)

    # --- Core numeric utilization ---
    if field_exists("time_in_hospital"):
        row["time_in_hospital"] = c1.number_input("Time in hospital (days)", 0, 30, 4)
    if field_exists("num_medications"):
        row["num_medications"] = c1.number_input("Number of medications", 0, 100, 10)
    if field_exists("number_inpatient"):
        row["number_inpatient"] = c2.number_input("Inpatient visits", 0, 30, 1)
    if field_exists("number_emergency"):
        row["number_emergency"] = c2.number_input("Emergency visits", 0, 30, 0)
    if field_exists("number_outpatient"):
        row["number_outpatient"] = c2.number_input("Outpatient visits", 0, 30, 0)
    if field_exists("number_diagnoses"):
        row["number_diagnoses"] = c3.number_input("Number of diagnoses", 0, 30, 6)

    # --- Engineered labs ---
    if field_exists("a1c_severity"):
        row["a1c_severity"] = c3.selectbox("A1C severity (0=Norm,1=>7,2=>8)", [0, 1, 2], index=0)
    if field_exists("glu_severity"):
        row["glu_severity"] = c3.selectbox("Glucose severity (0=Norm,1=>200,2=>300)", [0, 1, 2], index=0)

    # --- Binary / small categoricals ---
    if field_exists("diabetesMed"):
        row["diabetesMed"] = c1.selectbox("On diabetes meds?", ["No", "Yes"], index=1)
    if field_exists("change"):
        row["change"] = c2.selectbox("Medication change this encounter?", ["No", "Yes"], index=0)
    if field_exists("insulin"):
        row["insulin"] = c2.selectbox("Insulin status", ["No", "Up", "Down", "Steady"], index=0)
    if field_exists("gender"):
        row["gender"] = c3.selectbox("Gender", ["Male", "Female", "Unknown/Invalid"], index=1)
    if field_exists("race"):
        row["race"] = c3.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], index=0)
    if field_exists("admission_type_description"):
        row["admission_type_description"] = c3.selectbox(
            "Admission type",
            ["Emergency", "Elective", "Urgent", "Newborn", "Trauma Center", "Not Available"],
            index=0
        )

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    # Map simple flags to numeric if model was trained that way
    if "diabetesMed" in row:
        row["diabetesMed"] = 1 if row["diabetesMed"] == "Yes" else 0
    if "change" in row:
        row["change"] = 1 if row["change"] == "Yes" else 0

    # If the model expects age_mid but the user selected 'age' instead, derive midpoint
    if "age" in row and "age_mid" in expected_cols and "age_mid" not in row:
        row["age_mid"] = bracket_to_mid(row["age"])

    # Build DF and align
    X_row = pd.DataFrame([row])
    X_row = align_to_expected(X_row, expected_cols)

    # Predict
    try:
        proba = None
        try:
            proba = model.predict_proba(X_row)[0]
        except Exception:
            pass
        pred = model.predict(X_row)[0]
        st.success(f"**Prediction:** {pred}")

        if proba is not None and getattr(model, "classes_", None) is not None:
            labels = list(model.classes_)
            prob_df = pd.DataFrame({"Class": labels, "Probability": proba}) \
                        .sort_values("Probability", ascending=False).reset_index(drop=True)
            st.caption("Predicted class probabilities:")
            st.dataframe(prob_df, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
