import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('cancer_model.keras')

st.title("🧬 Cancer Detection App")
st.write("Upload a cell or medical image and let the model predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "🛑 Cancer Detected" if prediction[0][0] > 0.5 else "✅ No Cancer"
    st.subheader(result)
