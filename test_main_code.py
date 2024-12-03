import pytest
from fastapi import FastAPI
import common
from fastapi.testclient import TestClient
from main import app  # Importe ton application FastAPI

client = TestClient(app)


# Test de l'endpoint root (GET /)
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is ready to predict trip durations!"}
    print("Holla Sweety Maggy")


# Test de l'endpoint predict (POST /predict)
def test_predict_valid_input():
    # Données de tests valides
    data = {
        "weekday": 3,  # Mercredi
        "month": 5,  # Mai
        "hour": 14  # 14h
    }

    response = client.post("/predict", json=data)

    # Vérifie que la réponse est un dictionnaire et contient 'predicted_trip_duration'
    assert response.status_code == 200
    response_json = response.json()
    assert "predicted_trip_duration" in response_json
    assert isinstance(response_json["predicted_trip_duration"], (int, float))  # Vérifie que la prédiction est un nombre
    print("Maggy is very beautiful")

