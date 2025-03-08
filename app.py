import os
import streamlit as st
import joblib
import numpy as np
from db import insert_prediction  # Importing function from db.py
# Import MongoDB function

# Load models
models = {
    "SVM Binary": joblib.load('svm_binary.pkl'),
    "Logistic Regression (Binary)": joblib.load('logistics_binary.pkl'),
    "SVM Multiclass": joblib.load('svm_multi.pkl'),
    "Logistic Regression (OvR)": joblib.load('logistic_ovr.pkl'),
    "Logistic Regression (Multinomial)": joblib.load('logistics_multinomial.pkl')
}

# Load scalers (only for models that require scaling)
scalers = {
    "SVM Binary": joblib.load('svm_binary_scaler.pkl'),
    "Logistic Regression (Binary)": joblib.load('svm_binary_scaler.pkl')
}

# Image mapping for species
species_images = {
    "Setosa": "setosa.jpg",
    "Versicolor": "versicolor.jpeg",
    "Virginica": "virginica.jpeg"
}

# Streamlit UI
st.title('🌸 Iris Species Prediction')

# Sidebar for model selection
st.sidebar.header("🔍 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a Model for Prediction",
    list(models.keys())
)

# User input section
st.header("🌿 Enter Feature Values")

# User name input
user_name = st.text_input("Enter Your Name")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=3.5)
with col2:
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Prepare input data
original_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Apply scaling only if required
scaled_data = original_data.copy()
if model_choice in scalers:
    scaled_data = scalers[model_choice].transform(original_data)

# Predict button
if st.button('🔮 Predict Species'):
    if not user_name:
        st.warning("⚠️ Please enter your name before making a prediction.")
    else:
        # Load selected model
        selected_model = models[model_choice]

        # Make prediction
        prediction = selected_model.predict(scaled_data)

        # Map output to species names
        species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_species = species_map.get(prediction[0], 'Unknown')

        # Insert actual values (not scaled) into MongoDB
        insert_prediction(user_name, original_data, predicted_species, model_choice)

        # Display result
        st.subheader("🔹 Prediction Result")
        st.success(f"##### 🌼 Predicted Species: **{predicted_species}** using {model_choice}")
        st.info("📝 Your prediction has been saved in the database!")

        # Display the corresponding image
        image_path = species_images.get(predicted_species)
        if image_path:
            st.image(image_path, caption=f"🌸 {predicted_species}", width=200)
