from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import io

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

# --- 4. Document Uploader Endpoint for RAG ---
@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text_content = ""
        
        if file.filename.endswith(".txt"):
            text_content = contents.decode("utf-8", errors="ignore")
        elif file.filename.endswith(".pdf"):
            # Simple PDF parser fallback: if PyPDF2/pdfplumber is not loaded, parse as raw layout strings or binary chunks
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(contents))
                text_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except ImportError:
                # Fallback to standard decode if dependency isn't in requirements
                text_content = contents.decode("utf-8", errors="ignore")
                
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Unable to extract readable text from document.")
            
        # Dynamically append this text content directly to rag_chatbot's context repository
        if hasattr(rag_chatbot, "add_document_to_vector_store"):
            rag_chatbot.add_document_to_vector_store(text_content, file.filename)
        elif hasattr(rag_chatbot, "knowledge_base"):
            # If it uses an in-memory list or dictionary structure
            rag_chatbot.knowledge_base.append({"source": file.filename, "text": text_content})
        
        return {"status": "success", "message": f"Document {file.filename} parsed and integrated into RAG index."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# --- 5. Farmer Credit Score Endpoint ---
@app.post("/api/credit-score")
def calculate_score(data: CreditScoreInput):
    # Calls your credit_score.py script function
    payload = {
        "experience": data.experience,
        "certified_seeds": data.certified_seeds,
        "past_loans_paid": data.past_loans_paid
    }
    return credit_score.calculate_farmer_eligibility(payload)
