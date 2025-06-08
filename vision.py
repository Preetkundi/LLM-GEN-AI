from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

# 1) Page config must come first
st.set_page_config(
    page_title="Gemini Vision Demo",
    page_icon="🖼️",
    layout="centered",
)

st.title("🤖 Gemini Vision Q&A")

# 2) Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 3) Instantiate the vision model
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(text_prompt: str, img: Image.Image) -> str:
    """
    If text_prompt is non-empty, send both text+image;
    otherwise send image-only.
    """
    if text_prompt.strip():
        resp = model.generate_content([text_prompt, img])
    else:
        resp = model.generate_content(img)
    return resp.text

# 4) UI: prompt + uploader + submit
st.subheader("Ask Gemini about your image")
user_text = st.text_input("Enter a question (or leave blank to caption):")
uploaded = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="🖼️ Uploaded image", use_column_width=True)
else:
    img = None

if st.button("Submit"):
    if not img:
        st.error("Please upload an image first!")
    else:
        answer = get_gemini_response(user_text, img)
        st.subheader("Gemini says:")
        st.write(answer)
