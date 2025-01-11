import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('air_quality_model.h5')

# Streamlit app title and description
st.title("Air Pollution Detection from Sky Images")
st.write("Upload a sky image to predict air quality.")

# Upload image
uploaded_image = st.file_uploader("Choose a sky image...", type=["jpg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Map the predicted class to AQI label
    label_map = {0: "Good", 1: "Moderate", 2: "Unhealthy for Sensitive Groups", 
                 3: "Unhealthy", 4: "Very Unhealthy", 5: "Severe"}
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted Air Quality: {label_map[predicted_class]}")
    st.write(f"Confidence: {prediction[0][predicted_class]:.2f}")