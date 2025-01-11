import streamlit as st

st.title("Air & Wildfire Risk Detector")
st.write("Upload an image to get started!")

# File uploader
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing your image...")
    # Placeholder for AI model output
    st.write("Prediction: [Placeholder]")