# app.py  — FastAPI inference server for insurance premium category

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd

# -------- Load trained pipeline (model.pkl must be in the same folder) --------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------- US city tiers used during feature engineering --------
TIER_1_CITIES = {
    "New York","Los Angeles","Chicago","Houston","Phoenix",
    "Philadelphia","San Antonio","San Diego","Dallas","San Jose",
    "Austin","Jacksonville","Fort Worth","Columbus","Charlotte",
    "San Francisco","Indianapolis","Seattle","Denver","Washington",
    "Boston"
}
TIER_2_CITIES = {
    "Nashville","El Paso","Detroit","Oklahoma City","Portland",
    "Las Vegas","Memphis","Louisville","Baltimore","Milwaukee",
    "Albuquerque","Tucson","Fresno","Sacramento","Mesa",
    "Kansas City","Atlanta","Omaha","Colorado Springs","Raleigh",
    "Miami","Long Beach","Virginia Beach","Oakland","Minneapolis",
    "Tulsa","Arlington","Tampa","New Orleans","Wichita",
    "Cleveland","Bakersfield","Aurora","Anaheim","Honolulu",
    "Henderson","Riverside","Corpus Christi","Lexington",
    "Stockton","Hialeah","Anchorage","Plano","Greensboro"
}

# -------- Allowed occupations (same list used to train the model) --------
OccupationLiteral = Literal[
    "software_engineer","teacher","nurse","driver","sales_executive",
    "construction_worker","chef","artist","data_scientist","lawyer",
    "doctor","accountant","electrician","mechanic","retail_staff",
    "student","unemployed","business_owner","freelancer","government_job",
    "private_job","retired","researcher","warehouse_worker","security_guard"
]

# -------- Pydantic input schema (validated + computed fields) --------
class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age in years")]
    weight: Annotated[float, Field(..., gt=0, description="Weight in kg")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="Height in meters")]
    income_lpa: Annotated[float, Field(..., gt=0, description="Annual income in LPA")]
    smoker: Annotated[bool, Field(..., description="Is the person a smoker?")]
    city: Annotated[str, Field(..., description="US city")]
    occupation: Annotated[OccupationLiteral, Field(..., description="Occupation category")]

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in TIER_1_CITIES:
            return 1
        elif self.city in TIER_2_CITIES:
            return 2
        else:
            return 3


# -------- FastAPI app --------
app = FastAPI(title="Insurance Premium Classifier", version="1.0.0")


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "insurance-premium-api"}


@app.post("/predict", tags=["inference"])
def predict_premium(data: UserInput):
    """
    Returns: {"predicted_category": "Low" | "Medium" | "High"}
    """
    X = pd.DataFrame([{
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation
    }])
    prediction = model.predict(X)[0]
    return JSONResponse(status_code=200, content={"predicted_category": prediction})


# (Optional) probability endpoint if classifier supports predict_proba
@app.post("/predict_proba", tags=["inference"])
def predict_premium_proba(data: UserInput):
    """
    Returns class probabilities when available.
    """
    if not hasattr(model, "predict_proba"):
        return JSONResponse(status_code=400, content={"error": "Model does not support predict_proba"})

    X = pd.DataFrame([{
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation
    }])
    proba = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", ["Low", "Medium, ", "High"])
    return JSONResponse(status_code=200, content={"classes": list(map(str, classes)), "proba": proba.tolist()})
