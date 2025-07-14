"""Configuration module for the Hybrid Dense Reranker."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Corpus Configuration
CORPUS_SOURCE = os.getenv("CORPUS_SOURCE", "default")  # "default" or "mormon"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Overlap between chunks

# Application Configuration
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
FLASK_PORT = int(os.getenv("FLASK_PORT", "5002"))