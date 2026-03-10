import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from model import CNNModel


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")

st.write("Upload a handwritten digit image (0–9).")


# -----------------------------
# LOAD MODEL
# -----------------------------

model = CNNModel()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png","jpg","jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    # preprocess
    img = transform(image).unsqueeze(0)

    # prediction
    output = model(img)

    probs = torch.softmax(output,1).detach().numpy()[0]

    prediction = np.argmax(probs)

    confidence = probs[prediction]


    # confidence threshold
    threshold = 0.6


    with col2:

        st.subheader("Prediction Result")

        if confidence < threshold:

            st.error("⚠️ This image does not appear to be a handwritten digit.")

        else:

            st.success(f"Predicted Digit: {prediction}")

            st.write(f"Confidence: {confidence*100:.2f}%")

            # probability chart
            df = pd.DataFrame({
                "Digit": list(range(10)),
                "Probability": probs
            })

            st.subheader("Digit Probabilities")

            st.bar_chart(df.set_index("Digit"))