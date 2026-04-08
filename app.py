import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Satellite Oil Spill Detection System")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oil_spill_unet.h5", compile=False)

model = load_model()
IMG_SIZE = 256

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_mask(image):
    img = preprocess_image(image)
    prediction = model.predict(img)[0]
    mask = (prediction > 0.5).astype(np.uint8)
    return mask

def overlay_mask(image, mask):
    image = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    overlay = image.copy()

    # Highlight oil spill in red
    overlay[mask[:,:,0] == 1] = [255,0,0]

    return overlay


uploaded_file = st.file_uploader("Upload Satellite Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Satellite Image")
    st.image(image)

    mask = predict_mask(image)

    st.subheader("Predicted Oil Spill Mask")
    st.image(mask[:,:,0]*255)

    overlay = overlay_mask(image, mask)

    st.subheader("Oil Spill Highlighted on Image")
    st.image(overlay)

    # Calculate statistics
    oil_pixels = np.sum(mask)
    total_pixels = IMG_SIZE * IMG_SIZE

    oil_percentage = (oil_pixels / total_pixels) * 100

    st.subheader("Detection Statistics")

    st.write(f"Oil Spill Pixels Detected: {oil_pixels}")
    st.write(f"Total Pixels: {total_pixels}")
    st.write(f"Oil Spill Percentage: {oil_percentage:.2f}%")

    # Risk Level
    st.subheader("Risk Assessment")

    if oil_percentage < 1:
        st.success("No Oil Spill Detected")
        st.info("The satellite image does not show significant oil contamination.")
        st.write("Environmental Impact: Minimal or none.")

    elif oil_percentage < 5:
        st.warning("Minor Oil Spill Detected")
        st.write("Insight: Small patches of oil are visible in the water.")
        st.write("Environmental Impact: May affect small marine organisms locally.")

    elif oil_percentage < 15:
        st.warning("Moderate Oil Spill Detected")
        st.write("Insight: A noticeable oil spill area is present in the satellite image.")
        st.write("Environmental Impact: Possible impact on marine life and coastal ecosystems.")

    else:
        st.error("Severe Oil Spill Detected")
        st.write("Insight: Large oil contamination detected across the water surface.")
        st.write("Environmental Impact: High risk to marine biodiversity and nearby coastal areas.")
        st.write("Recommended Action: Immediate environmental monitoring and containment required.")
