import os
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- General Configuration ---
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    # Configure genai globally here if preferred, or within services
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google API Key loaded from environment variable.")
except ValueError as e:
    logger.error(f"ERROR: {e}")
    logger.error("Please set your GOOGLE_API_KEY environment variable. The application may not function correctly.")
    GOOGLE_API_KEY = None # Ensure it's None if not set, for checks elsewhere

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "models/text-embedding-004")
GENERATIVE_MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-1.5-flash") # Or "gemini-1.5-pro-latest"

# --- ChromaDB Configuration ---
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./vector_store/chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "application_guidelines_v1")

# --- Directory Configuration ---
# These are for default local storage if files are saved by backend,
# but primary interaction is via uploads.
GUIDELINE_PDF_DIR = "./data/guidance"
APPLICATION_PDF_DIR = "./data/pre_submitted_form"

# Ensure necessary directories exist (especially for ChromaDB)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
# os.makedirs(GUIDELINE_PDF_DIR, exist_ok=True) # Only if backend saves guidelines locally
# os.makedirs(APPLICATION_PDF_DIR, exist_ok=True) # Only if backend saves apps locally

# --- Text Chunking Configuration ---
CHARACTER_CHUNK_SIZE = 1000
CHARACTER_CHUNK_OVERLAP = 150
TOKEN_CHUNK_SIZE = 256
TOKEN_CHUNK_OVERLAP = 4

# --- API Configuration ---
API_TITLE = "RAG Application Validator API"
API_DESCRIPTION = "API for processing guidelines, validating applications, and RAG-based chat."
API_VERSION = "0.1.0"

# --- Retry Configuration for Google API calls ---
RETRY_CODES = [429, 500, 503] # HTTP codes that might trigger a retry

# --- Logging ---
def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)

logger = get_logger(__name__)
