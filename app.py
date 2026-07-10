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
    step=0.01
)

# iou_threshold = st.sidebar.slider(
#     "IoU Threshold",
#     min_value=0.10,
#     max_value=1.00,
#     value=0.50,
#     step=0.01
# )
iou_threshold = 0.5

from PIL import Image, ImageOps
import numpy as np

def roboflow_preprocess(image: Image.Image) -> Image.Image:
    """
    Approximate the preprocessing configured in the Roboflow dataset version:

    1. Auto-orient
    2. Stretch resize to 640 x 640
    3. Contrast stretching using the 2nd and 98th percentiles
    """

    # 1. Auto-orient using EXIF information
    image = ImageOps.exif_transpose(image).convert("RGB")

    # 2. Stretch directly to 640 x 640
    image = image.resize(
        (640, 640),
        resample=Image.Resampling.BILINEAR
    )

    image_np = np.asarray(image).astype(np.float32)

    # 3. Contrast stretching
    # Process each RGB channel independently
    stretched = np.empty_like(image_np)

    for channel in range(3):
        channel_data = image_np[:, :, channel]

        lower = np.percentile(channel_data, 2)
        upper = np.percentile(channel_data, 98)

        if upper > lower:
            channel_data = (
                (channel_data - lower)
                / (upper - lower)
                * 255.0
            )

        stretched[:, :, channel] = np.clip(
            channel_data,
            0,
            255
        )

    return Image.fromarray(stretched.astype(np.uint8))

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a wall image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    processed_image = roboflow_preprocess(
        original_image
    )

    processed_np = np.asarray(processed_image)

    with st.spinner("Detecting wall cracks..."):

        results = model.predict(
            source=processed_np,
            imgsz=640,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )

    annotated = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            original_image,
            use_container_width=True
        )

    with col2:
        st.subheader("Detection Result")
        st.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

    # with st.expander("View preprocessed model input"):
    #     st.image(
    #         processed_image,
    #         caption=(
    #             "Auto-oriented, stretched to 640 × 640, "
    #             "and contrast-stretched"
    #         ),
    #         use_container_width=True
    #     )

    boxes = results[0].boxes

    st.divider()
    st.subheader("Detection Summary")

    if boxes is None or len(boxes) == 0:

        st.warning(
            "No wall crack was detected at the selected "
            "confidence threshold."
        )

    else:

        # confidences = [
        #     float(box.conf.item())
        #     for box in boxes
        # ]

        # st.metric(
        #     "Detected Crack Regions",
        #     len(confidences)
        # )

        # st.metric(
        #     "Highest Confidence",
        #     f"{max(confidences):.3f}"
        # )
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