
from predictor import load_model, predict_price

# Charger votre modèle
model = load_model()

# Exemple avec les valeurs de votre notebook
result = predict_price(
    model=model,
    longitude=-122,
    latitude=37.24,
    housing_age=33,
    total_rooms=2840,
    total_bedrooms=560,
    population=1890,
    households=489,
    median_income=7.14,
    ocean_proximity="INLAND",
)

print(f"Valeur prédite : ${result:,.2f}")
