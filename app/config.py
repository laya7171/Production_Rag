import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    """Application settings"""
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROK_API_KEY = os.getenv("XAI_API_KEY")
    QDRANT_COLLECTION_NAME = "Production-Rag"
    GEMINI_MODEL = "gemini-3-flash-preview"
    GROK_MODEL = "llama-3.3-70b-versatile"



settings = Settings()