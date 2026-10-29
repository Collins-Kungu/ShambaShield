from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importing your custom service scripts exactly as named:
import weather_alert_service
import maize_price_trend
import rag_chatbot
import credit_score

app = FastAPI(title="SmartMaize Backend API", version="1.0")

# --- Schemas ---
class CreditScoreInput(BaseModel):
    experience: int
    certified_seeds: bool
    past_loans_paid: bool

class ChatInput(BaseModel):
    query: str

# --- 1. Weather Alert Endpoint ---
@app.get("/api/weather")
def get_weather(location: str = "Nakuru"):
    # Calls your weather_alert_service.py script function
    return weather_alert_service.get_weather_alerts(location)

# --- 2. Price Trend Endpoint ---
@app.get("/api/price-trend")
def get_price_trend():
    # Calls your maize_price_trend.py script function
    return maize_price_trend.get_historical_trends()

# --- 3. Chatbot & RAG Endpoint ---
@app.post("/api/chat")
def chat_advisor(data: ChatInput):
    # Calls your rag_chatbot.py script function
    return rag_chatbot.query_vector_store(data.query)

# --- 4. Farmer Credit Score Endpoint ---
@app.post("/api/credit-score")
def calculate_score(data: CreditScoreInput):
    # Calls your credit_score.py script function
    payload = {
        "experience": data.experience,
        "certified_seeds": data.certified_seeds,
        "past_loans_paid": data.past_loans_paid
    }
    return credit_score.calculate_farmer_eligibility(payload)
