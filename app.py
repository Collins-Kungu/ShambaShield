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
        st.session_state.rag_messages.append(
            {"role": "user", "content": user_query}
        )
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
        st.session_state.rag_messages.append(
            {"role": "assistant", "content": response}
        )

# =====================================================================
# 5. Farmer credit score.ipynb (Custom Weighted System Implementation)
# =====================================================================
elif service == "💰 Farmer Credit Score":
    st.title("💰 Farmer Credit Score & Loan Waitlist")
    st.subheader("Agricultural Risk Assessment Platform")
    st.write("Evaluate loan qualification instantly using core field metrics and soil indices.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Demographics & Basic Information")
        farmer_id = st.text_input("Farmer Registration ID", "F-9824")
        farmer_name = st.text_input("Farmer Full Name", "Jane Atieno")
        farm_size = st.number_input("Size of Farm (Hectares)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
        
    with col2:
        st.subheader("🌱 Scientific & Risk Parameters")
        soil_moisture = st.slider("Soil Moisture Content (%)", 0.0, 100.0, 35.0, step=0.5)
        ndvi = st.slider("Normalized Difference Vegetation Index (NDVI)", 0.0, 1.0, 0.65, step=0.01)
        climate_risk = st.selectbox("Climate Risk Zone", ["Low", "Medium", "High"])

    if st.button("Run Weighted Credit Evaluation"):
        if not farmer_id or not farmer_name:
            st.error("Please make sure both Farmer ID and Farmer Name are provided before evaluating credit risk.")
        else:
            # Define appropriate mathematical weights corresponding to variable importance
            # Total potential score = 100
            # 1. Soil Moisture Contribution (Max: 30 Points)
            # Optimal moisture target is between 25% and 60%
            if 25.0 <= soil_moisture <= 60.0:
                moisture_score = 30.0
            elif 15.0 <= soil_moisture < 25.0 or 60.0 < soil_moisture <= 80.0:
                moisture_score = 15.0
            else:
                moisture_score = 5.0

            # 2. NDVI Contribution (Max: 40 Points)
            # Higher vegetation density represents higher crop health and yield safety
            ndvi_score = ndvi * 40.0

            # 3. Farm Size Contribution (Max: 15 Points)
            # Moderate scaled land offers optimal commercial risk diversification
            if farm_size >= 5.0:
                size_score = 15.0
            elif 1.5 <= farm_size < 5.0:
                size_score = 10.0
            else:
                size_score = 5.0

            # 4. Climate Risk Mitigation Contribution (Max: 15 Points)
            if climate_risk == "Low":
                risk_score = 15.0
            elif climate_risk == "Medium":
                risk_score = 8.0
            else:
                risk_score = 2.0

            # Calculate final combined weighted score
            total_score = moisture_score + ndvi_score + size_score + risk_score
            passed_eligibility = total_score >= 55.0

            st.markdown("---")
            st.subheader(f"Evaluation Report: {farmer_name} ({farmer_id})")
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Computed Credit Index Score", f"{total_score:.1f} / 100")
            
            with metric_col2:
                if passed_eligibility:
                    st.success("STATUS: LOAN WAITLIST APPROVED ✔️")
                else:
                    st.error("STATUS: WAITLIST ON HOLD ✖️")

            # Interactive breakdown analysis of weighted factors
            st.markdown("### Weighted Parameter Breakdown:")
            breakdown_data = {
                "Parameter": ["Soil Moisture", "NDVI Index", "Farm Size (Ha)", "Climate Risk Zone"],
                "Assigned Weight": ["30%", "40%", "15%", "15%"],
                "User Input Metric": [f"{soil_moisture}%", f"{ndvi}", f"{farm_size} Ha", climate_risk],
                "Weighted Points Earned": [f"{moisture_score:.1f} / 30", f"{ndvi_score:.1f} / 40", f"{size_score:.1f} / 15", f"{risk_score:.1f} / 15"]
            }
            st.table(pd.DataFrame(breakdown_data))

            if passed_eligibility:
                st.balloons()
                st.info(f"Congratulations {farmer_name}! Your agronomic metrics confirm a sustainable risk profile. You have been queued to the ShambaShapeUp farmer micro-credit program waitlist.")
            else:
                st.warning(f"The computed index of {total_score:.1f} falls below the baseline qualification margin of 55.0 points. Advise the farmer on field modifications, moisture conservation techniques, or fertilizer inputs to optimize canopy NDVI for future re-application.")
