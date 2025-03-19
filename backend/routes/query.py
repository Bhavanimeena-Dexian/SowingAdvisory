from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.gpt import generate_gpt_response 

# Create a FastAPI router
query_router = APIRouter()

# Define request model
class QueryRequest(BaseModel):
    query: str

@query_router.post("/")
async def handle_query(request: QueryRequest):
    """
    API endpoint to process user queries.

    Expects JSON:
    {
        "query": "User's question"
    }

    Returns:
        JSON response containing the answer.
    """
    user_query = request.query

    if not user_query:
        raise HTTPException(status_code=400, detail="Missing query parameter")

    # Generate response
    answer = generate_gpt_response(user_query)
    return {"query": user_query, "answer": answer}
