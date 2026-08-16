import cv2
import streamlit as st
import numpy as np


def face_detection():

    st.title("👤 Face Detection")

    st.write(
        "Capture an image using your webcam or upload an image "
        "to detect faces using OpenCV Haar Cascade."
    )

    # ---------------------------------------------------------
    # Load Haar Cascade
    # ---------------------------------------------------------
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        st.error("❌ Face detector could not be loaded.")
        return

    st.success("✅ Face detector loaded")

    # ---------------------------------------------------------
    # Choose input method
    # ---------------------------------------------------------
    input_method = st.radio(
        "Choose input method:",
        ["📷 Use Webcam", "📁 Upload Image"],
        horizontal=True
    )

    image_bytes = None

    # ---------------------------------------------------------
    # Webcam
    # ---------------------------------------------------------
    if input_method == "📷 Use Webcam":

        st.subheader("📷 Capture a Photo")

        camera_photo = st.camera_input(
            "Take a picture"
        )

        if camera_photo is not None:
            image_bytes = camera_photo.getvalue()

    # ---------------------------------------------------------
    # Upload Image
    # ---------------------------------------------------------
    else:

        st.subheader("📁 Upload an Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()

    # ---------------------------------------------------------
    # Face Detection
    # ---------------------------------------------------------
    if image_bytes is not None:

        # Convert bytes to NumPy array
        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        # Read image using OpenCV
        face_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if face_image is None:
            st.error("❌ Could not read the image.")
            return

        # Convert to grayscale
        gray_image = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Copy image for drawing
        output_image = face_image.copy()

        # Draw rectangles
        for (x, y, w, h) in faces:

            cv2.rectangle(
                output_image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        # Convert BGR → RGB
        output_rgb = cv2.cvtColor(
            output_image,
            cv2.COLOR_BGR2RGB
        )

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------

        st.divider()

        st.subheader(
            f"🎯 Detected Faces: {len(faces)}"
        )

        st.image(
            output_rgb,
            caption=f"Detected Faces: {len(faces)}",
            use_container_width=True
        )

        if len(faces) > 0:

            st.success(
                f"✅ {len(faces)} face(s) detected!"
            )

        else:

            st.warning(
                "⚠️ No faces detected."
            )