from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_PATH = Path("titanic_model.pkl")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]

app = FastAPI(title="Titanic Survival Prediction API")
model = None
model_accuracy = None


class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., ge=0)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str


def build_model() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def train_and_save_model() -> tuple[Pipeline, float]:
    print("Loading data...")
    df = pd.read_csv(DATA_URL)

    X = df[FEATURES]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    trained_model = build_model()
    trained_model.fit(X_train, y_train)

    predictions = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(trained_model, MODEL_PATH)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Saved model to {MODEL_PATH}")

    return trained_model, accuracy


@app.get("/")
def root():
    return {
        "message": "Titanic model API is running",
        "model_file": str(MODEL_PATH),
        "accuracy": model_accuracy,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict_survival(passenger: Passenger):
    passenger_data = pd.DataFrame([passenger.model_dump()], columns=FEATURES)
    prediction = int(model.predict(passenger_data)[0])
    probability = float(model.predict_proba(passenger_data)[0][prediction])

    return {
        "survived": bool(prediction),
        "prediction": prediction,
        "probability": round(probability, 4),
    }


if __name__ == "__main__":
    model, model_accuracy = train_and_save_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
