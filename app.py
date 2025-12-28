import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from datetime import datetime

from src.model import get_model

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="Jackson or Not Jackson",
    layout="centered"
)

st.image("assets/banner.png", use_container_width=True)
st.markdown("<h3 style='color:white;'>Made by Cholorsplash </h3>", unsafe_allow_html=True)


st.markdown("""

### Context
This project explores CNN models to classify images and determine whether they represent **Jackson Wang** or not.

### Problematic
Jackson Wang is sometimes mistaken for other idols, which can frustrate fans.
This application uses deep learning to assist with identification.

### Who is Jackson Wang?
Jackson Wang is a Hong Kong rapper, singer, and performer, member of GOT7.
He has a successful international solo career and is also active in fashion and entertainment.
""")


# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name="optimized", device=device)
    model.load_state_dict(torch.load("models/model_best.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# =====================
# Image preprocessing
# =====================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =====================
# Image upload
# =====================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    # Save uploaded image in session_state to persist between reruns
    st.session_state["uploaded_image"] = image

    if st.button("Analyse"):
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        class_names = ["Not Jackson", "Jackson"]
        predicted_class = class_names[prediction.item()]
        confidence_score = confidence.item() * 100

        st.markdown(f"### Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence_score:.2f}%")

        # Store results in session_state
        st.session_state["prediction"] = predicted_class
        st.session_state["confidence"] = confidence_score

        # =====================
        # User feedback
        # =====================
        st.markdown("### Is this prediction correct?")
        col1, col2 = st.columns(2)

        feedback = None
        if col1.button("✅ Yes", key="yes"):
            feedback = True
        if col2.button("❌ No", key="no"):
            feedback = False

        # if feedback is not None:
        #     os.makedirs("feedback/images", exist_ok=True)
        #     feedback_path = "feedback/feedback.csv"

        #     # Save uploaded image
        #     timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     image_filename = f"feedback/images/{timestamp_str}.png"
        #     st.session_state["uploaded_image"].save(image_filename)

        #     row = {
        #         "timestamp": timestamp_str,
        #         "prediction": st.session_state["prediction"],
        #         "confidence": st.session_state["confidence"],
        #         "user_feedback": feedback,
        #         "image_path": image_filename
        #     }

        #     df = pd.DataFrame([row])

        #     if os.path.exists(feedback_path):
        #         df.to_csv(feedback_path, mode="a", header=False, index=False)
        #     else:
        #         df.to_csv(feedback_path, index=False)

            st.success("Thank you for your feedback! 🙏")
