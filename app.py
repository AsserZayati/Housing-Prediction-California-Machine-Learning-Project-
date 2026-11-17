
import streamlit as st

from predictor import load_model, predict_price


# Configuration de l'application
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


st.title("🏠 California Housing Predictor")

st.write(
    "Estimez la valeur immobilière médiane "
    "d'un secteur résidentiel en Californie."
)

st.info(
    "Les informations demandées concernent "
    "un secteur géographique, et non "
    "une maison individuelle."
)


# Charger le modèle
try:
    model = get_model()

except Exception as error:
    st.error(f"Impossible de charger le modèle : {error}")
    st.stop()


# Formulaire de prédiction
with st.form("housing_form"):

    st.subheader("📍 Localisation du secteur")

    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-113.0,
            value=-121.50,
            format="%.4f",
            help="Coordonnée géographique ouest-est.",
        )

    with col2:
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=43.0,
            value=37.50,
            format="%.4f",
            help="Coordonnée géographique nord-sud.",
        )

    st.subheader("🏘️ Caractéristiques des logements")

    col3, col4 = st.columns(2)

    with col3:
        housing_age = st.number_input(
            "Âge médian des logements (années)",
            min_value=0,
            max_value=100,
            value=25,
        )

        total_rooms = st.number_input(
            "Nombre total de pièces dans le secteur",
            min_value=1,
            value=2400,
        )

        total_bedrooms = st.number_input(
            "Nombre total de chambres dans le secteur",
            min_value=0,
            value=500,
        )

    with col4:
        population = st.number_input(
            "Population du secteur",
            min_value=0,
            value=1600,
        )

        households = st.number_input(
            "Nombre de ménages dans le secteur",
            min_value=1,
            value=600,
        )

        median_income = st.number_input(
            "Revenu médian du secteur (× 10 000 $)",
            min_value=0.0,
            value=4.2,
            step=0.1,
            help=(
                "Par exemple, 4.2 correspond à une "
                "valeur de revenu de 42 000 $ "
                "dans l'échelle du dataset."
            ),
        )

    st.subheader("🌊 Situation par rapport à l'océan")

    ocean_options = {
        "À l'intérieur des terres": "INLAND",
        "À moins d'une heure de l'océan": "<1H OCEAN",
        "Près de l'océan": "NEAR OCEAN",
        "Près de la baie de San Francisco": "NEAR BAY",
        "Sur une île": "ISLAND",
    }

    ocean_label = st.selectbox(
        "Quelle description correspond au secteur ?",
        options=list(ocean_options.keys()),
    )

    submitted = st.form_submit_button(
        "🔮 Estimer la valeur immobilière",
        use_container_width=True,
    )


# Exécution de la prédiction
if submitted:

    try:
        result = predict_price(
            model=model,
            longitude=longitude,
            latitude=latitude,
            housing_age=housing_age,
            total_rooms=total_rooms,
            total_bedrooms=total_bedrooms,
            population=population,
            households=households,
            median_income=median_income,
            ocean_proximity=ocean_options[ocean_label],
        )

        st.success("Prédiction effectuée !")

        st.metric(
            label="Valeur immobilière médiane estimée",
            value=f"${result:,.2f}",
        )

        st.caption(
            "Estimation statistique du modèle. "
            "Ce résultat n'est pas une expertise "
            "immobilière individuelle."
        )

    except ValueError as error:
        st.warning(str(error))

    except Exception as error:
        st.error(f"Erreur de prédiction : {error}")
