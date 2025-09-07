US Insurance Premium Classifier 

We trained a scikit-learn model on US insurance data, engineered features (BMI, age_group, lifestyle_risk, city_tier), and saved the fitted pipeline as model.pkl.
Then we served predictions via a FastAPI /predict endpoint and built a Streamlit UI that posts user inputs to the API and shows the predicted premium (Low/Medium/High).

Predict: Low / Medium / High premium using a trained scikit-learn pipeline served by FastAPI with a Streamlit UI.


├─ frontend.py                 # FastAPI server (loads model.pkl)
├─ streamlit_app.py            # Streamlit UI (calls FastAPI)
├─ model.pkl                   # Trained sklearn Pipeline
├─ Insurance_Premium_Data.csv  # 200-row training data (US cities)
|-FastApi_MLmodel #Machine learning Code(Training)
└─ README.md



Files:

frontend.py – FastAPI server (loads model.pkl, exposes /predict, /predict_proba)
streamlit_app.py – Streamlit UI (calls FastAPI)
model.pkl – trained pipeline
Insurance_Premium_Data.csv – sample training data

#How to Run the Code:

conda activate C:\Users\Nitish\Desktop\Coding\FastAPI\venv_test_fastapi

uvicorn app:app --reload

streamlit run frontend.py

Example request (POST /predict)
{
  "age": 34,
  "weight": 82.0,
  "height": 1.78,
  "income_lpa": 18.2,
  "smoker": false,
  "city": "Seattle",
  "occupation": "software_engineer"
}
Response: {"predicted_category": "Low"}

Tech stack:
--------------
Python 3.10+
scikit-learn, pandas
FastAPI, Uvicorn
Pydantic v2 (for request validation & computed fields)
Streamlit

