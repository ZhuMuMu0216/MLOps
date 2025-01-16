import streamlit as st
import requests
from PIL import Image

# 后端 URL
BACKEND_URL = "http://localhost:8000/predict/"  # 替换为后端部署的 Cloud Run URL

# 前端界面
st.title("Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 显示用户上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # 点击按钮发送请求到后端
    if st.button("Classify"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(BACKEND_URL, files=files)
        
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Prediction: {prediction['prediction']['class']} (Probability: {prediction['prediction']['probability']:.2f})")
        else:
            st.error("Error: Unable to get prediction from backend.")
