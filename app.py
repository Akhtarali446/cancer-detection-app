import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load model once and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('1.keras')

model = load_model()

# Streamlit UI
st.title("🧬 Cancer Detection App")
st.write("Upload a cell or medical image. The AI will predict whether cancer is present.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and prepare the image
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)

    st.image(image, caption='🖼️ Uploaded Image', use_container_width=True)

    # Preprocess image
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    with tf.device("/CPU:0"):  # Change to "/GPU:0" if available and desired
        prediction = model.predict(img_array, verbose=0)

    confidence = float(prediction[0][0])
    result = "🛑 **Cancer Detected**" if confidence > 0.5 else "✅ **No Cancer**"

    # Display prediction results
    st.subheader(result)
    st.metric(label="Prediction Confidence", value=f"{confidence:.2%}")

    # Optional: Display threshold guidance
    st.caption("Predictions above 50% are considered positive for cancer.")
