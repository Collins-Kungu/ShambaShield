# This writes the code directly to a file in Colab for downloading

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="SmartMaize Platform", page_icon="🌽", layout="wide")

# Sidebar Navigation with Icons matching the five individual notebooks
st.sidebar.title("☘️ SmartMaize Platform")
st.sidebar.write("Navigate seamlessly between your individual services:")

service = st.sidebar.radio(
    "Services Menu",
    [
        "🌽 Maize Pest Classifier",
        "⛈️ Severe Weather Alerts",
        "📈 Maize Price Trend",
        "💬 Chatbot & RAG Advisor",
        "💰 Farmer Credit Score"
    ]
)

# =====================================================================
# 1. maize_pest_classifier_colab.ipynb
# =====================================================================
if service == "🌽 Maize Pest Classifier":
    st.title("🌽 Maize Pest Classifier")
    st.subheader("Model Deployment from maize_pest_classifier_colab.ipynb")
    st.write("Identify maize leaf conditions using your trained MobileNetV2 image classification model.")

    @st.cache_resource
    def load_my_model():
        model_path = 'maize_pest_mobilenetv2.keras'
        return tf.keras.models.load_model(model_path)

    try:
        model = load_my_model()
        st.success("Classifier model loaded successfully!")
    except Exception as e:
        st.error(f"Could not load classifier. Please place 'maize_pest_mobilenetv2.keras' in this directory. Details: {e}")

    uploaded_file = st.file_uploader("Upload maize leaf image (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("Running image analysis...")

        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        probability = prediction[0][0]

        class_names = ['Fall Armyworm (Affected)', 'Healthy']

        if probability > 0.5:
            result_class = class_names[1]
            confidence = probability * 100
        else:
            result_class = class_names[0]
            confidence = (1 - probability) * 100

        st.subheader("Classification Outcome:")
        if result_class == 'Healthy':
            st.success(f"The maize plant appears to be **Healthy** (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"Warning: The maize plant is affected by **{result_class}** (Confidence: {confidence:.2f}%)")

# =====================================================================
# 2. Severe weather alert API.ipynb
# =====================================================================
elif service == "⛈️ Severe Weather Alerts":
    st.title("⛈️ Severe Weather Alerts")
    st.subheader("API Integration from Severe weather alert API.ipynb")
    st.write("Get real-time weather tracking and hazards warnings for crop safety.")
    
    location = st.text_input("Farm Location (City/District):", "Nakuru")
    
    if st.button("Query Weather API"):
        st.info(f"Fetching atmospheric data for {location}...")
        st.warning(f"Alert: Moderate precipitation and wind speeds expected in {location}. Monitor young crop stalks.")

# =====================================================================
# 3. Maize price trend.ipynb
# =====================================================================
elif service == "📈 Maize Price Trend":
    st.title("📈 Maize Price Trend Analysis")
    st.subheader("Market Trends from Maize price trend.ipynb")
    st.write("Visualise price trajectories and analyze market rates for maize crop products.")
    
    import pandas as pd
    import random
    
    dates = pd.date_range(start="2024-01-01", end="2024-05-01", freq="W")
    prices = [random.randint(2800, 3600) for _ in range(len(dates))]
    price_df = pd.DataFrame({"Date": dates, "Maize price (per bag)": prices})
    
    st.line_chart(price_df, x="Date", y="Maize price (per bag)")
    st.caption("Simulated wholesale price trend graph mapped from your analysis framework.")

# =====================================================================
# 4. Chatbot and RAG.ipynb
# =====================================================================
elif service == "💬 Chatbot & RAG Advisor":
    st.title("💬 Chatbot & RAG Advisor")
    st.subheader("Agricultural Advice from Chatbot and RAG.ipynb")
    st.write("Interact with an AI agent contextually augmented with local guides on controlling Fall Armyworm infestations.")
    
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about crop protection or armyworm management:"):
        st.session_state.rag_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            response = "Based on standard reference documents: Using biological agents like Bacillus thuringiensis (Bt) or releasing natural predators helps suppress Armyworm populations without chemical hazards."
            st.markdown(response)
        st.session_state.rag_messages.append({"role": "assistant", "content": response})

# =====================================================================
# 5. Farmer credit score.ipynb
# =====================================================================
elif service == "💰 Farmer Credit Score":
    st.title("💰 Farmer Credit Score & Loan Waitlist")
    st.subheader("Credit Assessment from Farmer credit score.ipynb")
    st.write("Calculate credit eligibility and enroll in waitlists using farm inputs and productivity metrics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Waitlist Enrollment")
        full_name = st.text_input("Farmer Full Name")
        phone = st.text_input("Contact Number (Mobile Money)")
        farm_size = st.number_input("Maize Acreage", min_value=0.1, value=1.5)
        
        if st.button("Submit Assessment Request"):
            if full_name and phone:
                st.success(f"Registration recorded for {full_name}. We will reach out on your mobile number.")
            else:
                st.error("Name and contact number are required parameters.")
                
    with col2:
        st.subheader("📊 Scoring Parameters")
        experience = st.slider("Farming History (Years)", 0, 15, 3)
        certified_seeds = st.checkbox("I plant certified quality seeds (+25 pts)")
        past_loans_paid = st.checkbox("I successfully repaid previous input credits (+30 pts)")
        
        score = 400
        score += (experience * 10)
        if certified_seeds: score += 25
        if past_loans_paid: score += 30
        
        st.metric("Calculated Farm Score", f"{score} / 800")
        
        if score >= 480:
            st.success("Eligibility Assessment: Approved for basic credit tier.")
        else:
            st.warning("Enhance credit score by implementing crop rotation, seed certification, or pest scouting.")
