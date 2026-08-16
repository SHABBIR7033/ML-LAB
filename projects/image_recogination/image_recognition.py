import time

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

# st.set_page_config(
#     page_title="Image Recognition",
#     page_icon="🖼️",
#     layout="centered",
# )


@st.cache_resource(show_spinner="Loading pretrained MobileNetV2 model (first run downloads ~14 MB)...")
def load_model():
    """Load the ImageNet-pretrained MobileNetV2 model once and cache it."""
    return MobileNetV2(weights="imagenet")


def predict(model, image: Image.Image, top: int = 5):
    """Run inference on a PIL image and return the top-N decoded predictions."""
    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)

    start_time = time.time()
    predictions = model.predict(processed_image, verbose=0)
    elapsed_ms = (time.time() - start_time) * 1000

    decoded = decode_predictions(predictions, top=top)[0]
    return decoded, elapsed_ms


def image_recognition():
    st.title("🖼️ Image Recognition")
    st.write(
        "Upload an image and this app will classify it using **MobileNetV2**, "
        "a convolutional neural network pretrained on the ImageNet dataset "
        "(1,000 everyday object categories)."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            decoded_predictions, elapsed_ms = predict(model, image)

        with col2:
            st.subheader("Top predictions")
            for _, label, probability in decoded_predictions:
                label = label.replace("_", " ").title()
                st.write(f"**{label}**")
                st.progress(float(probability))
                st.caption(f"{probability * 100:.2f}%")

        st.caption(f"Prediction took {elapsed_ms:.1f} ms")
    else:
        st.info("👆 Upload an image to get started.")



