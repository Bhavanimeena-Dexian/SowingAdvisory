import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fetch credentials securely
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Ensure all required variables are set
if not ENDPOINT_URL or not DEPLOYMENT_NAME or not AZURE_OPENAI_API_KEY:
    raise ValueError("❌ Missing environment variables! Check .env file.")
