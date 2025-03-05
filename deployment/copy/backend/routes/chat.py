from fastapi import APIRouter
from pydantic import BaseModel

# Define the FastAPI router
chat_router = APIRouter()

# Request model to ensure valid input
class ChatRequest(BaseModel):
    message: str

# Response model (Optional, but good for OpenAPI docs)
class ChatResponse(BaseModel):
    reply: str

# API endpoint to handle chat messages
@chat_router.post("/", response_model=ChatResponse)
async def get_chat_response(request: ChatRequest):
    # Placeholder response (Later, replace with an LLM API call)
    bot_reply = f"🌱 Smart Advisory: Based on your input '{request.message}', here is some farming advice! 🌿"
    
    return {"reply": bot_reply}
