"""
app.py
------
Streamlit web app for Handwritten Digit & Number Sequence Recognition.

Features:
  1. Draw digits/numbers on an interactive canvas (horizontal & vertical reading order)
  2. Upload an image of single or multi-digit handwritten numbers
  3. Predicts number sequences ordered Top-to-Bottom and Left-to-Right
  4. Provides line-by-line digit previews and confidence breakdowns.

Author: Naravula Prudhvi Sri Bhanu Vivek
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

from utils import segment_and_preprocess_image, pil_from_canvas_rgba

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Number Sequence Recognizer",
    page_icon="🔢",
    layout="centered",
)

st.title("🔢 Handwritten Number Sequence Recognizer")
st.caption("Powered by a CNN trained on MNIST & Automated Top-to-Bottom, Left-to-Right Reading Order Segmentation")


# ---------------------------------------------------------------
# Load model once, cache it across reruns
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "Model file `digit_model.keras` not found. "
        "Please run `python train_model.py` first to train and save the model."
    )


def predict_sequence(pil_image: Image.Image):
    """
    Extract top-to-bottom, left-to-right digit segments, run CNN prediction for each,
    and return (sequence_string, digits_info_list).
    """
    segments = segment_and_preprocess_image(pil_image)
    if not segments:
        return None, []

    digits_info = []
    for item in segments:
        probs = model.predict(item["tensor"], verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        digits_info.append({
            "digit": digit,
            "confidence": confidence,
            "probs": probs,
            "preview_pil": item["preview_pil"],
            "bbox": item["bbox"],
            "line_idx": item.get("line_idx", 0),
        })

    # Group by line_idx
    lines_dict = {}
    for d in digits_info:
        l_idx = d["line_idx"]
        lines_dict.setdefault(l_idx, []).append(d)

    line_strings = ["".join(str(d["digit"]) for d in digits) for digits in lines_dict.values()]
    if len(line_strings) > 1:
        line_formatted = [f"Line {idx + 1}: {s}" for idx, s in enumerate(line_strings)]
        sequence_str = "  |  ".join(line_formatted)
    else:
        sequence_str = line_strings[0] if line_strings else ""

    return sequence_str, digits_info


def show_sequence_result(sequence_str: str, digits_info: list):
    if not sequence_str or not digits_info:
        st.warning("No digits detected in the image. Please draw clearly or upload a clearer image.")
        return

    st.success(f"### Predicted Reading Sequence: **{sequence_str}**")
    st.metric(
        label="Predicted Number / Sequence",
        value=sequence_str,
        delta=f"{len(digits_info)} digit(s) detected",
    )

    st.subheader("🔍 Line-by-Line Digit Breakdown")
    st.caption("Segments detected in reading order (Top-to-Bottom, Left-to-Right):")

    # Group digits by line
    lines_dict = {}
    for info in digits_info:
        lines_dict.setdefault(info["line_idx"], []).append(info)

    for line_order_idx, (line_idx, line_digits) in enumerate(lines_dict.items()):
        line_val = "".join(str(d["digit"]) for d in line_digits)
        st.markdown(f"### 📍 Line {line_order_idx + 1}: **{line_val}**")

        num_digits = len(line_digits)
        cols = st.columns(min(num_digits, 6))

        for idx, info in enumerate(line_digits):
            col = cols[idx % min(num_digits, 6)]
            with col:
                st.markdown(f"**Digit #{idx + 1}**")
                st.image(info["preview_pil"], caption="28x28 crop (MNIST)", width=90)
                st.markdown(f"### **{info['digit']}**")
                st.caption(f"Conf: **{info['confidence'] * 100:.1f}%**")
                with st.expander("Probabilities"):
                    st.bar_chart({"Prob": info["probs"]}, height=140)


# ---------------------------------------------------------------
# Landing screen: choose an input method
# ---------------------------------------------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None

if st.session_state.mode is None:
    st.subheader("How would you like to provide numbers?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✍️ Write Now", use_container_width=True):
            st.session_state.mode = "draw"
            st.rerun()
    with col2:
        if st.button("📤 Upload an Image", use_container_width=True):
            st.session_state.mode = "upload"
            st.rerun()

else:
    if st.button("← Back to options"):
        st.session_state.mode = None
        st.rerun()

    st.divider()

    # -----------------------------------------------------------
    # Mode 1: Draw on a wide pad
    # -----------------------------------------------------------
    if st.session_state.mode == "draw":
        st.subheader("✍️ Draw digits / number sequence below")
        st.caption("Draw digits horizontally or vertically (Top-to-Bottom, Left-to-Right) with clear spacing.")

        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=16,
            stroke_color="white",
            background_color="black",
            height=320,
            width=560,
            drawing_mode="freedraw",
            key="canvas",
        )

        col1, col2 = st.columns(2)
        with col1:
            predict_clicked = st.button("🔍 Predict Sequence", use_container_width=True,
                                         disabled=not model_loaded)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()

        if predict_clicked:
            if canvas_result.image_data is None or canvas_result.image_data[:, :, 3].max() == 0:
                st.warning("Please draw at least one digit first.")
            else:
                pil_img = pil_from_canvas_rgba(canvas_result.image_data)
                sequence_str, digits_info = predict_sequence(pil_img)
                show_sequence_result(sequence_str, digits_info)

    # -----------------------------------------------------------
    # Mode 2: Upload an image
    # -----------------------------------------------------------
    elif st.session_state.mode == "upload":
        st.subheader("📤 Upload an image of handwritten digits")
        st.caption("Upload a photo or scan of single, multi-digit, or multi-line handwritten numbers.")

        uploaded_file = st.file_uploader(
            "Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Uploaded Image", width=350)

            if st.button("🔍 Predict Sequence", disabled=not model_loaded):
                sequence_str, digits_info = predict_sequence(pil_img)
                show_sequence_result(sequence_str, digits_info)

st.divider()
st.caption("Built by Naravula Prudhvi Sri Bhanu Vivek — MNIST CNN Digit & Sequence Recognizer")


