import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd

# Configure local URL pointing to your running FastAPI backend server
API_BASE_URL = "http://127.0.0.1:8000/api"

# Set page title and favicon
st.set_page_config(page_title="SmartMaize Platform", page_icon="ሔ", layout="wide")

# Sidebar Navigation with Icons matching the five individual notebooks
st.sidebar.title("☘ SmartMaize Platform")
st.sidebar.write("Navigate seamlessly between your individual services:")

service = st.sidebar.radio(
    "Services Menu",
    [
        "ሔ Maize Pest Classifier",
        "☔ Severe Weather Alerts",
        "ለ Maize Price Trend",
        "ሳ Chatbot & RAG Advisor",
        "ቁ Farmer Credit Score"
    ]
)

# =====================================================================
# 1. maize_pest_classifier_colab.ipynb (Kept local for high throughput)
# =====================================================================
if service == "ሔ Maize Pest Classifier":
    st.title("ሔ Maize Pest Classifier")
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
# 2. Severe weather alert API.ipynb (Direct Live Open-Meteo Integration)
# =====================================================================
elif service == "☔ Severe Weather Alerts":
    st.title("☔ Severe Weather Alerts")
    st.subheader("Live Meteorological Data from Open-Meteo API")
    st.write("Get real-time weather tracking and hazards warnings directly from the frontend interface.")

    location = st.text_input("Farm Location (City/District):", "Nakuru")

    if st.button("Query Live Weather"):
        st.info(f"Searching geocoding coordinates for {location}...")
        try:
            # Fetch Coordinates for named city
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_res = requests.get(geo_url)
            if geo_res.status_code == 200 and geo_res.json().get("results"):
                result = geo_res.json()["results"][0]
                lat = result["latitude"]
                lon = result["longitude"]
                city_name = result["name"]
                country = result.get("country", "")
                
                st.success(f"Located: {city_name}, {country} (Lat: {lat}, Lon: {lon})")
                
                # Fetch weather metrics
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
                weather_res = requests.get(weather_url)
                
                if weather_res.status_code == 200:
                    w_data = weather_res.json()["current"]
                    temp = w_data["temperature_2m"]
                    humidity = w_data["relative_humidity_2m"]
                    wind = w_data["wind_speed_10m"]
                    code = w_data["weather_code"]
                    
                    # Interpret warnings based on values
                    alerts = []
                    if temp > 35.0:
                        alerts.append("ሔ Extreme Heat Warning: Ensure proper irrigation to protect young crops.")
                    if wind > 25.0:
                        alerts.append("ሡ High Wind Advisory: Physical damage or lodging risk for mature maize.")
                    if code in [95, 96, 99]:
                        alerts.append("ሣ Severe Thunderstorms detected in current forecast.")
                        
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Temperature", f"{temp} ሓC")
                    with col2:
                        st.metric("Humidity", f"{humidity} %")
                    with col3:
                        st.metric("Wind Speed", f"{wind} km/h")
                        
                    st.subheader("☔ Field Warnings")
                    if alerts:
                        for alert in alerts:
                            st.warning(alert)
                    else:
                        st.success("No immediate severe weather hazards or crop risks detected.")
                else:
                    st.error("Unable to retrieve meteorological metrics from Open-Meteo.")
            else:
                st.error(f"Could not find location coordinates for '{location}'. Please verify the spelling.")
        except Exception as e:
            st.error(f"Network or validation error occurred: {e}")

# =====================================================================
# 3. Maize price trend.ipynb (Delegated to maize_price_trend.py)
# =====================================================================
elif service == "ለ Maize Price Trend":
    st.title("ለ Maize Price Trend Analysis")
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
elif service == "ሳ Chatbot & RAG Advisor":
    st.title("ሳ Chatbot & RAG Advisor")
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
elif service == "ቁ Farmer Credit Score":
    st.title("ቁ Farmer Credit Score & Loan Waitlist")
    st.subheader("Agricultural Risk Assessment Platform")
    st.write("Evaluate loan qualification instantly using core field metrics and soil indices.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ሳ Demographics & Basic Information")
        farmer_id = st.text_input("Farmer Registration ID", "F-9824")
        farmer_name = st.text_input("Farmer Full Name", "Jane Atieno")
        farm_size = st.number_input("Size of Farm (Hectares)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)

    with col2:
        st.subheader("ሱ Scientific & Risk Parameters")
        soil_moisture = st.slider("Soil Moisture Content (%)", 0.0, 100.0, 35.0, step=0.5)
        ndvi = st.slider("Normalized Difference Vegetation Index (NDVI)", 0.0, 1.0, 0.65, step=0.01)
        climate_risk = st.selectbox("Climate Risk Zone", ["Low", "Medium", "High"])

    if st.button("Run Weighted Credit Evaluation"):
        if not farmer_id or not farmer_name:
            st.error("Please make sure both Farmer ID and Farmer Name are provided before evaluating credit risk.")
        else:
            # 1. Soil Moisture Contribution (Max: 30 Points)
            if 25.0 <= soil_moisture <= 60.0:
                moisture_score = 30.0
            elif 15.0 <= soil_moisture < 25.0 or 60.0 < soil_moisture <= 80.0:
                moisture_score = 15.0
            else:
                moisture_score = 5.0

            # 2. NDVI Contribution (Max: 40 Points)
            ndvi_score = ndvi * 40.0

            # 3. Farm Size Contribution (Max: 15 Points)
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
                    st.success("STATUS: LOAN WAITLIST APPROVED ᄕ")
                else:
                    st.error("STATUS: WAITLIST ON HOLD ᄖ")

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
