from pathlib import Path
import pickle
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, computed_field


MODEL_PATH = Path(__file__).resolve().with_name("Model.pkl")
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

app = FastAPI(title="Insurance Premium Category Predictor", version="1.0.0")

TIER_1_CITIES = {
    "mumbai", "delhi", "bengaluru", "bangalore", "chennai", "kolkata", "hyderabad", "pune"
}
TIER_2_CITIES = {
    "jaipur", "chandigarh", "indore", "lucknow", "patna", "ranchi",
    "visakhapatnam", "coimbatore", "bhopal", "nagpur", "vadodara", "surat",
    "jodhpur", "raipur", "amritsar", "varanasi", "agra", "dehradun", "mysuru",
    "mysore", "jabalpur", "guwahati", "thiruvananthapuram", "ludhiana", "nashik",
    "prayagraj", "allahabad", "udaipur", "aurangabad", "hubballi", "hubli",
    "belagavi", "belgaum", "salem", "vijayawada", "tiruchirappalli", "bhavnagar",
    "gwalior", "dhanbad", "bareilly", "aligarh", "gaya", "kozhikode", "warangal",
    "kolhapur", "bilaspur", "jalandhar", "noida", "guntur", "asansol", "siliguri"
}


class UserInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    age: Annotated[int, Field(gt=0, lt=120, description="Age in years")]
    weight: Annotated[float, Field(gt=0, le=500, description="Weight in kg")]
    height: Annotated[float, Field(gt=0, le=2.5, description="Height in meters")]
    income_lpa: Annotated[float, Field(gt=0, le=100_000, description="Annual income in lakh rupees")]
    smoker: bool
    city: Annotated[str, Field(min_length=1, max_length=100)]
    occupation: Literal[
        "Government Employee", "Doctor", "Business", "Accountant", "Teacher",
        "Other", "Student", "Software Developer", "Manager", "Engineer"
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        if self.smoker and self.bmi > 27:
            return "medium"
        return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "Young"
        if self.age < 45:
            return "Adult"
        if self.age < 60:
            return "middle_aged"
        return "Senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        city = self.city.casefold()
        if city in TIER_1_CITIES:
            return 1
        if city in TIER_2_CITIES:
            return 2
        return 3


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Insurance predictor API is running"}


@app.post("/predict")
def predict_premium(data: UserInput):
    model_input = pd.DataFrame([{
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
    }])

    predicted_category = str(model.predict(model_input)[0])
    result = {
        "bmi": data.bmi,
        "lifestyle_risk": data.lifestyle_risk,
        "age_group": data.age_group,
        "city_tier": data.city_tier,
        "predicted_category": predicted_category,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        result["category_probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        } 
    return result
