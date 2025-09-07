# streamlit_app.py  — US Insurance Premium Classifier (matches our training data)

import streamlit as st
import requests

# --- Page config & title ---
st.set_page_config(page_title="US Insurance Premium Classifier", page_icon="💼", layout="centered")
st.title("US Insurance Premium Classifier")
st.markdown("Enter your details below to predict **Low / Medium / High** premium category.")

# --- API endpoint (change if your FastAPI runs elsewhere) ---
API_URL = st.text_input("FastAPI Predict Endpoint", value="http://34.226.152.222:8000/predict", help="Example: http://host:port/predict")
API_PROBA_URL = API_URL.replace("/predict", "/predict_proba")

# --- Allowed options per our dataset/training ---
US_CITIES = [
    "New York","Los Angeles","Chicago","Houston","Phoenix",
    "Philadelphia","San Antonio","San Diego","Dallas","San Jose",
    "Austin","Jacksonville","Fort Worth","Columbus","Charlotte",
    "San Francisco","Indianapolis","Seattle","Denver","Washington",
    "Boston","Nashville","El Paso","Detroit","Oklahoma City",
    "Portland","Las Vegas","Memphis","Louisville","Baltimore"
]

OCCUPATIONS = [
    "software_engineer","teacher","nurse","driver","sales_executive",
    "construction_worker","chef","artist","data_scientist","lawyer",
    "doctor","accountant","electrician","mechanic","retail_staff",
    "student","unemployed","business_owner","freelancer","government_job",
    "private_job","retired","researcher","warehouse_worker","security_guard"
]

# --- Inputs (aligned to model features) ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=119, value=34)
    weight = st.number_input("Weight (kg)", min_value=1.0, value=82.0, step=0.1)
    height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.78, step=0.01)
with col2:
    income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=18.2, step=0.1)
    smoker = st.selectbox("Are you a smoker?", options=[False, True], index=0)
    city = st.selectbox("City (US)", options=sorted(US_CITIES), index=US_CITIES.index("Seattle") if "Seattle" in US_CITIES else 0)

occupation = st.selectbox("Occupation", options=sorted(OCCUPATIONS), index=sorted(OCCUPATIONS).index("software_engineer"))

want_proba = st.toggle("Also fetch class probabilities (if server supports /predict_proba)", value=False)

# --- Predict button ---
if st.button("Predict Premium Category", type="primary"):
    if not API_URL.strip():
        st.error("Please provide a valid FastAPI endpoint URL.")
    else:
        input_data = {
            "age": int(age),
            "weight": float(weight),
            "height": float(height),
            "income_lpa": float(income_lpa),
            "smoker": bool(smoker),
            "city": city,
            "occupation": occupation
        }

        try:
            # Primary prediction
            resp = requests.post(API_URL, json=input_data, timeout=15)
            result = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200:
                # Our FastAPI returns {"predicted_category": "..."}
                if "predicted_category" in result:
                    st.success(f"Predicted Insurance Premium Category: **{result['predicted_category']}**")
                # Fallback to alternate shape {"response": {"predicted_category": "...", ...}}
                elif "response" in result and isinstance(result["response"], dict) and "predicted_category" in result["response"]:
                    st.success(f"Predicted Insurance Premium Category: **{result['response']['predicted_category']}**")
                else:
                    st.info("Received response:")
                    st.json(result)
            else:
                st.error(f"API Error: {resp.status_code}")
                st.write(result)

            # Optional probabilities
            if want_proba:
                try:
                    proba_resp = requests.post(API_PROBA_URL, json=input_data, timeout=15)
                    proba_json = proba_resp.json()
                    if proba_resp.status_code == 200 and isinstance(proba_json, dict):
                        st.subheader("Class Probabilities")
                        st.json(proba_json)
                    else:
                        st.warning("Predict_proba endpoint not available or returned an error.")
                except requests.exceptions.RequestException:
                    st.warning("Could not call /predict_proba (endpoint may not exist).")

        except requests.exceptions.ConnectionError:
            st.error(" Could not connect to the FastAPI server. Make sure it's running and reachable.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")
