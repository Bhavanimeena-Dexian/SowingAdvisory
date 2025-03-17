from fastapi import APIRouter
from pydantic import BaseModel
from services.gpt import generate_gpt_response  # Import GPT response function

# Define FastAPI router
chat_router = APIRouter()

# Request model
class ChatRequest(BaseModel):
    message: str

# Response model
class ChatResponse(BaseModel):
    reply: str

# API endpoint to handle chat messages
@chat_router.post("/", response_model=ChatResponse)
async def get_chat_response(request: ChatRequest):
    """
    Processes user query and fetches GPT-4o response.
    """
    bot_reply = generate_gpt_response(request.message)  # Calls GPT function
    return {"reply": bot_reply}
