from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveInt
import pickle
import os
import numpy as np
import common
import uvicorn

# Charger le modèle depuis le fichier
model = common.load_model(common.MODEL_PATH)

# Création de l'API FastAPI
app = FastAPI()


# Modèle Pydantic pour valider les entrées
class PredictionInput(BaseModel):
    weekday: PositiveInt = Field(..., ge=1, le=7, description="Day of the week (1=Monday, ..., 7=Sunday)")
    month: PositiveInt = Field(..., ge=1, le=12, description="Month of the year (1=January, ..., 12=December)")
    hour: PositiveInt = Field(..., ge=0, le=23, description="Hour of the day (0 to 23)")

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Endpoint pour prédire la durée du trajet.
    """
    # Convertir les données d'entrée en format compatible avec le modèle
    features = np.array([[input_data.weekday, input_data.month, input_data.hour]])

    # Prédiction
    prediction = model.predict(features)[0]

    return {"predicted_trip_duration": prediction}


@app.get("/")
def root():
    return {"message": "API is ready to predict trip durations!"}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8080, reload=True)


