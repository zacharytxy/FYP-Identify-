import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

# Initialize the Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8vYYsvtFfCX0nzLiY3PZ"
)

# Page settings
st.set_page_config(
    page_title="License Plate & Container ID Detector",
    layout="centered",
    page_icon="📸"
)

# Header
st.markdown("<h1 style='text-align: center;'>📸 License Plate & Container ID Detector</h1>", unsafe_allow_html=True)
st.write("Upload an image and we’ll detect license plates or container IDs using your trained Roboflow model.")

# File upload
uploaded_file = st.file_uploader("### 📂 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save and display image
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_path = "uploaded_image.jpg"
    image = Image.open(image_path)
    st.image(image_path, use_container_width=True, caption="Uploaded Image")

    # Inference
    with st.spinner("🔍 Running detection..."):
        result = CLIENT.infer(image_path, model_id="custom-workflow-object-detection-i7df3/3")

    st.success("✅ Detection Complete!")

    predictions = result.get("predictions", [])
    
    if not predictions:
        st.warning("No objects detected.")
    else:
        for i, pred in enumerate(predictions):
            st.markdown(f"### 🔎 Detected Object #{i+1}")
            st.markdown(
                f"""
                - **Class:** `{pred['class'].replace('_', ' ').title()}`
                - **Confidence:** `{round(pred['confidence'] * 100, 2)}%`
                - **Coordinates:** X = {int(pred['x'])}, Y = {int(pred['y'])}
                - **Box Size:** Width = {int(pred['width'])}, Height = {int(pred['height'])}
                - **Detection ID:** `{pred['detection_id']}`
                """)
            
        # Optional note
        st.info("""
        **What do the values mean?**
        - **Class**: The label the model assigned (e.g., license plate).
        - **Confidence**: How sure the model is about the prediction. A higher % means higher certainty.
        - **Coordinates & Box Size**: Where the object was found in the image.
        """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Roboflow")

