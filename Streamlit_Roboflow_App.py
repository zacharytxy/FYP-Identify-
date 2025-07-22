import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8vYYsvtFfCX0nzLiY3PZ"
)

# Page config
st.set_page_config(page_title="License Plate & Container ID Detector", page_icon="📸")

st.markdown("<h1 style='text-align: center;'>📸 License Plate & Container ID Detector</h1>", unsafe_allow_html=True)
st.write("Upload an image or use your webcam to detect license plates or container IDs using your trained Roboflow model.")

# Choose input method
option = st.radio("Choose image source:", ["📁 Upload an image", "📷 Take a photo"])

image = None

# Upload
if option == "📁 Upload an image":
    uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Camera
elif option == "📷 Take a photo":
    captured = st.camera_input("Take a picture")
    if captured:
        image = Image.open(captured)
        st.image(image, caption="Captured Image", use_container_width=True)

# Inference
if image:
    st.info("Click below to detect license plates or container IDs.")
    if st.button("🚀 Run Detection"):
        # Save to disk for inference
        image_path = "temp_image.jpg"
        image.save(image_path)

        with st.spinner("Detecting..."):
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
                    - **Class:** {pred['class'].replace('_', ' ').title()}
                    - **Confidence:** {round(pred['confidence'] * 100, 2)}%
                    - **Coordinates:** X = {int(pred['x'])}, Y = {int(pred['y'])}
                    - **Box Size:** Width = {int(pred['width'])}, Height = {int(pred['height'])}
                    - **Detection ID:** {pred['detection_id']}
                    """
                )

            st.info("""
            **Metrics Explanation:**
            - **Class**: Type of object (e.g. license plate).
            - **Confidence**: Model certainty (closer to 100% is better).
            - **Box Size**: Size of the detected area.
            """)

# Show model training insights
#with st.expander("📈 Model Training Insights"):
    #st.markdown("These charts represent the training progress of the YOLOv8 model used in this app:")
    
    #st.image("images/box_loss.png", caption="Box Loss - Measures the accuracy of bounding box predictions.")
    #st.image("images/class_loss.png", caption="Class Loss - Measures how well the model classified objects.")
    #st.image("images/object_loss.png", caption="Object Loss - Tracks how confidently the model detects the object.")
    #st.image("images/performance_map.png", caption="Model Performance (mAP) - Higher values indicate better detection accuracy across all classes.")

