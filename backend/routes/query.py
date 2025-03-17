from flask import Blueprint, request, jsonify
from services.gpt import generate_gpt_response

# Create a Blueprint for handling queries
query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def handle_query():
    """
    API endpoint to process user queries.

    Expects JSON:
    {
        "query": "User's question"
    }

    Returns:
        JSON response containing the answer.
    """
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Missing query parameter"}), 400

    # Generate response
    answer = generate_gpt_response(user_query)
    return jsonify({"query": user_query, "answer": answer})
