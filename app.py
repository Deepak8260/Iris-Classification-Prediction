import streamlit as st
import joblib
import numpy as np

# Load all models
models = {
    "SVM Binary": joblib.load('svm_binary.pkl'),
    "Logistic Regression (Binary)": joblib.load('logistics_binary.pkl'),
    "SVM Multiclass": joblib.load('svm_multi.pkl'),
    "Logistic Regression (OvR)": joblib.load('logistic_ovr.pkl'),
    "Logistic Regression (Multinomial)": joblib.load('logistics_multinomial.pkl')
}

# Load the scaler (used only for some models)
scaler = joblib.load('svm_binary_scaler.pkl')

# Models that require scaling
models_requiring_scaling = ["SVM Binary", "Logistic Regression (Binary)"]

# Title of the app
st.title('🌸 Iris Species Prediction')

# Sidebar for model selection
st.sidebar.header("🔍 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a Model for Prediction",
    list(models.keys())
)

# Main Section for Inputs
st.header("🌿 Enter Feature Values")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=3.5)

with col2:
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Prepare input data
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Apply scaling only if the selected model requires it
if model_choice in models_requiring_scaling:
    input_data = scaler.transform(input_data)

# Predict when the button is clicked
if st.button('🔮 Predict Species'):
    # Load the selected model
    selected_model = models[model_choice]

    # Make prediction
    prediction = selected_model.predict(input_data)

    # Mapping output to species names
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_map.get(prediction[0], 'Unknown')

    # Display result
    st.subheader("🔹 Prediction Result")
    st.success(f" 🌼 Predicted Species: **{predicted_species}** using {model_choice}")
