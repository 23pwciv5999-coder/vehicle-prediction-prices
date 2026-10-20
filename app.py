
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Price Prediction")
st.write("Enter the details of a vehicle to estimate its price.")

MODEL_PATH = Path(__file__).parent / "model_bundle.pkl"
bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
options = bundle["options"]

numeric_defaults = bundle["numeric_defaults"]
numeric_min = bundle["numeric_min"]
numeric_max = bundle["numeric_max"]

with st.form("vehicle_form"):
    st.subheader("Vehicle Details")
    user_input = {}

    for col in numeric_cols:
        minimum = float(numeric_min[col])
        maximum = float(numeric_max[col])
        default = float(numeric_defaults[col])

        if minimum == maximum:
            user_input[col] = minimum
        else:
            default = min(max(default, minimum), maximum)
            user_input[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1.0
            )

    for col in categorical_cols:
        choices = options.get(col, ["unknown"]) or ["unknown"]
        default_index = (
            choices.index("unknown") if "unknown" in choices else 0
        )

        user_input[col] = st.selectbox(
            col.replace("_", " ").title(),
            choices,
            index=default_index
        )

    submitted = st.form_submit_button("Predict Vehicle Price")

if submitted:
    input_df = pd.DataFrame([user_input])
    input_df = input_df[bundle["feature_columns"]]

    for col in categorical_cols:
        input_df[col] = (
            input_df[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    prediction = pipeline.predict(input_df)[0]

    st.success("Prediction completed!")
    st.metric("Estimated Vehicle Price", f"{prediction:,.0f}")
    st.caption(
        "This is an estimate based on historical training data, "
        "not a guaranteed selling price."
    )
