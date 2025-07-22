import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

# Initialize the Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8vYYsvtFfCX0nzLiY3PZ"
)

# Streamlit UI
st.set_page_config(page_title="License Plate & Container ID Detector")
st.title("📸 License Plate & Container ID Detector")
st.markdown("Upload an image and we'll detect license plates or container IDs using your Roboflow model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_path = "uploaded_image.jpg"

    # Show the uploaded image
    image = Image.open(image_path)
    st.image(image_path, use_container_width=True)


    # Run inference
    with st.spinner("Detecting..."):
        result = CLIENT.infer(image_path, model_id="custom-workflow-object-detection-i7df3/3")

    # Show result
    st.success("Detection Complete!")
    st.json(result)
