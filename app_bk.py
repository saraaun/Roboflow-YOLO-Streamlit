import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from utils.detector import load_model

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Wall Crack Detection",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Wall Crack Detection")
st.write(
    "Upload a wall image to detect visible cracks using a YOLO model trained with Roboflow."
)

# ----------------------------------------------------
# Load model
# ----------------------------------------------------

model = load_model()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

# iou_threshold = st.sidebar.slider(
#     "IoU Threshold",
#     min_value=0.10,
#     max_value=1.00,
#     value=0.45,
#     step=0.05
# )
iou_threshold = 0.5

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a wall image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # image = Image.open(uploaded_file).convert("RGB")

    from PIL import Image, ImageOps
    image = Image.open(uploaded_file)
    # Auto-Orient
    image = ImageOps.exif_transpose(image)
    # Convert to RGB
    image = image.convert("RGB")
    # Contrast Stretching
    image = ImageOps.autocontrast(image)
    image_np = np.array(image)

    with st.spinner("Detecting wall cracks..."):

        results = model.predict(
            image_np,
            imgsz=640,
            conf=confidence,
            iou= iou_threshold,
            verbose=False
        )

    annotated = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(annotated, use_container_width=True)

    boxes = results[0].boxes

    st.divider()

    st.subheader("Detection Summary")

    if len(boxes) == 0:

        st.success("✅ No wall cracks detected.")

    else:

        confidences = []

        for box in boxes:
            confidences.append(float(box.conf))

        df = pd.DataFrame({
            "Crack ID": range(1, len(confidences)+1),
            "Confidence": [round(c,3) for c in confidences]
        })

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Detected Cracks",
                len(confidences)
            )

        with metric2:
            st.metric(
                "Average Confidence",
                f"{np.mean(confidences):.2f}"
            )

        st.dataframe(
            df,
            use_container_width=True
        )