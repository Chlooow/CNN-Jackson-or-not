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

st.title("Jackson or Not Jackson?")
st.subheader("CNN-based Image Classification")

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

    if st.button("Analyse"):
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        class_names = ["Not Jackson", "Jackson"]
        predicted_class = class_names[prediction.item()]
        confidence_score = confidence.item() * 100

        st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence_score:.2f}%")

        # =====================
        # User feedback
        # =====================
        st.markdown("### Is this prediction correct?")
        col1, col2 = st.columns(2)

        feedback = None
        if col1.button("✅ Yes"):
            feedback = True
        if col2.button("❌ No"):
            feedback = False

        if feedback is not None:
            os.makedirs("feedback", exist_ok=True)
            feedback_path = "feedback/feedback.csv"

            row = {
                "timestamp": datetime.now(),
                "prediction": predicted_class,
                "confidence": confidence_score,
                "user_feedback": feedback
            }

            df = pd.DataFrame([row])

            if os.path.exists(feedback_path):
                df.to_csv(feedback_path, mode="a", header=False, index=False)
            else:
                df.to_csv(feedback_path, index=False)

            st.success("Thank you for your feedback! 🙏")
