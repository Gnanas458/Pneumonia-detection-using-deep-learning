import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Page title
st.title("Pneumonia Detection from Chest X-Ray Images")
st.markdown("Upload a chest X-ray image to predict whether the patient has Pneumonia.")

# Load model
@st.cache_resource
def load_cnn_model():
    model = load_model(r"D:\Final_project\Pneumonia Detection from Chest X-Ray Images using CNN\xray_model.h5")
    return model

model = load_cnn_model()
class_names = ["Normal", "Pneumonia"]

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((150, 150))  # Replace with your model's input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.write("### Prediction:")
    st.success(f"**{class_names[predicted_class]}** with {confidence*100:.2f}% confidence")
