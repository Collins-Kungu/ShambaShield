import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd

# Configure local URL pointing to your running FastAPI backend server
API_BASE_URL = "http://127.0.0.1:8000/api"

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
# 1. maize_pest_classifier_colab.ipynb (Kept local for high throughput)
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
# 2. Severe weather alert API.ipynb (Delegated to weather_alert_service.py)
# =====================================================================
elif service == "⛈️ Severe Weather Alerts":
    st.title("⛈️ Severe Weather Alerts")
    st.subheader("API Integration from weather_alert_service.py")
    st.write("Get real-time weather tracking and hazards warnings for crop safety.")

    location = st.text_input("Farm Location (City/District):", "Nakuru")

    if st.button("Query Weather API"):
        st.info(f"Querying FastAPI backend for {location}...")
        try:
            res = requests.get(f"{API_BASE_URL}/weather", params={"location": location})
            if res.status_code == 200:
                data = res.json()
                st.success(f"Successfully pulled weather metrics for {data.get('location', location)}")
                st.metric("Temperature", data.get("temperature", "N/A"))
                st.metric("Humidity", data.get("humidity", "N/A"))
                st.warning(data.get("alert", "No alerts found."))
            else:
                st.error("Backend returned an error responding to this query.")
        except Exception as e:
            st.error(f"Could not reach FastAPI backend server. Ensure it is running. Error: {e}")

# =====================================================================
# 3. Maize price trend.ipynb (Delegated to maize_price_trend.py)
# =====================================================================
elif service == "📈 Maize Price Trend":
    st.title("📈 Maize Price Trend Analysis")
    st.subheader("Market Trends from maize_price_trend.py")
    st.write("Visualise price trajectories and analyze market rates for maize crop products.")

    if st.button("Fetch Market Analysis"):
        try:
            res = requests.get(f"{API_BASE_URL}/price-trend")
            if res.status_code == 200:
                data = res.json()
                df = pd.DataFrame({
                    "Date": pd.to_datetime(data["dates"]),
                    "Maize price (per bag)": data["prices"]
                })
                st.line_chart(df, x="Date", y="Maize price (per bag)")
                st.success("Real-time analytics chart rendered successfully from the backend database.")
            else:
                st.error("Could not fetch data from market service endpoints.")
        except Exception as e:
            st.error(f"Could not reach backend. Error details: {e}")

# =====================================================================
# 4. Chatbot and RAG.ipynb (Delegated to rag_chatbot.py)
# =====================================================================
elif service == "💬 Chatbot & RAG Advisor":
    st.title("💬 Chatbot & RAG Advisor")
    st.subheader("Agricultural Advice from rag_chatbot.py")
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
        
        try:
            res = requests.post(f"{API_BASE_URL}/chat", json={"query": user_query})
            if res.status_code == 200:
                response = res.json().get("response", "No reply context.")
            else:
                response = "Failed to retrieve context from knowledge base. Check backend logs."
        except Exception as e:
            response = f"Error communicating with RAG Microservice API: {e}"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.rag_messages.append({"role": "assistant", "content": response})

# =====================================================================
# 5. Farmer credit score.ipynb (Delegated to credit_score.py)
# =====================================================================
elif service == "💰 Farmer Credit Score":
    st.title("💰 Farmer Credit Score & Loan Waitlist")
    st.subheader("Credit Assessment from credit_score.py")
    st.write("Calculate credit eligibility and enroll in waitlists using farm inputs and productivity metrics.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Waitlist Enrollment")
        full_name = st.text_input("Farmer Full Name")
        phone = st.text_input("Contact Number (Mobile Money)")
        farm_size = st.number_input("Maize Acreage", min_value=0.1, value=1.5)
        
        if st.button("Submit Assessment Request"):
            if full_name and phone:
                st.success(f"Registration recorded for {full_name}. We will run verification processes.")
            else:
                st.error("Name and contact number are required parameters.")
                
    with col2:
        st.subheader("📊 Scoring Parameters")
        experience = st.slider("Farming History (Years)", 0, 15, 3)
        certified_seeds = st.checkbox("I plant certified quality seeds (+25 pts)")
        past_loans_paid = st.checkbox("I successfully repaid previous input credits (+30 pts)")
        
        if st.button("Calculate Eligibility"):
            try:
                payload = {
                    "experience": experience,
                    "certified_seeds": certified_seeds,
                    "past_loans_paid": repaid if 'repaid' in locals() else past_loans_paid
                }
                res = requests.post(f"{API_BASE_URL}/credit-score", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.metric("Calculated Farm Score", f"{data.get('score', 0)} / 800")
                    if data.get("eligible", False):
                        st.success(data.get("recommendation", "Approved"))
                    else:
                        st.warning(data.get("recommendation", "Needs Improvement"))
                else:
                    st.error("Could not verify credit metrics from backend services.")
            except Exception as e:
                st.error(f"Credit microservice offline. Details: {e}")
