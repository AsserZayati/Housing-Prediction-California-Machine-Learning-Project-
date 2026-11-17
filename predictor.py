
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Chemin vers votre modèle entraîné
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "modele_maison_prices.joblib"
)

CATEGORIES_OCEAN = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]


def load_model():
    """Charger le modèle Random Forest sauvegardé."""
    return joblib.load(MODEL_PATH)


def prepare_data(
    longitude,
    latitude,
    housing_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity,
    model,
):
    """Préparer les variables comme dans Code.ipynb."""

    # Données brutes saisies par l'utilisateur
    data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])

    # Vérifications des valeurs
    if total_rooms <= 0 or households <= 0:
        raise ValueError(
            "Le nombre de pièces et de ménages "
            "doit être supérieur à zéro."
        )

    if total_bedrooms > total_rooms:
        raise ValueError(
            "Le nombre de chambres ne peut pas "
            "dépasser le nombre total de pièces."
        )

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude invalide.")

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude invalide.")

    # 1. Création des caractéristiques
    # Comme dans votre notebook
    data["bedroom_ratio"] = (
        data["total_bedrooms"] / data["total_rooms"]
    )

    data["household_rooms"] = (
        data["total_rooms"] / data["households"]
    )

    # 2. Transformations logarithmiques
    # Identiques à celles de votre notebook
    for column in [
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "household_rooms",
    ]:
        data[column] = np.log(data[column] + 1)

    # 3. Encodage de la proximité de l'océan
    data["ocean_proximity"] = pd.Categorical(
        data["ocean_proximity"],
        categories=CATEGORIES_OCEAN,
    )

    data = pd.get_dummies(
        data,
        columns=["ocean_proximity"],
    )

    # 4. Ordre exact des colonnes du modèle
    data = data[model.feature_names_in_]

    return data


def predict_price(model, **values):
    """Retourner la valeur immobilière prédite."""

    data = prepare_data(model=model, **values)

    prediction = model.predict(data)[0]

    return float(prediction)
