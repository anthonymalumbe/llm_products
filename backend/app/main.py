from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse

from app.api import endpoints as api_endpoints
from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, get_logger, GENERATIVE_MODEL_NAME

# # from backend.app.api import endpoints as api_endpoints
# # from backend.app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, get_logger, GENERATIVE_MODEL_NAME

# # from backend.app.services.vector_store_service import vector_store_service_instance # Ensure it's initialized
# # from backend.app.services.document_processor_service import document_processor_instance

# Ensure llm_service is accessible for the lifespan event
# Option 1: Direct import if no circular dependency or pre-init issues
# from app.services.llm_service import generative_model # (See note in lifespan)

# Initialize logger
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """
    Manages startup and shutdown events for the FastAPI application.
    """
    logger.info("FastAPI application startup commencing...")
    
    # Ensure LLM service is ready
    # Local import here can be useful if the service itself does complex initialization
    # or to avoid top-level import issues before everything is set up.
    try:
        from app.services.llm_service import generative_model # Assuming this is the correct path
        if generative_model:
            logger.info(f"LLM service with model '{GENERATIVE_MODEL_NAME}' is ready.")
        else:
            # This else might be hit if 'generative_model' is None after import
            logger.error("LLM service initialized, but generative_model is None (check GOOGLE_API_KEY and service logic).")
    except ImportError:
        logger.error("Failed to import llm_service.generative_model. Ensure the service path is correct and dependencies are met.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM service initialization: {e}")

    # Example for vector store check (if you uncomment its usage)
    try:
        from app.services.vector_store_service import vector_store_service_instance
        if vector_store_service_instance and vector_store_service_instance.client and vector_store_service_instance.collection:
            collection_name = getattr(vector_store_service_instance, 'collection_name', 'N/A')
            collection_count = -1 # Default or placeholder
            # The method get_collection_count might not exist or might fail if the collection isn't fully ready
            try:
                collection_count = vector_store_service_instance.get_collection_count()
            except Exception as e_count:
                logger.warning(f"Could not get collection count for '{collection_name}': {e_count}")
            logger.info(f"Vector store service connected. Collection '{collection_name}' count: {collection_count}")
        else:
            logger.error("Vector store service failed to initialize properly during startup.")
    except ImportError:
        logger.error("Failed to import vector_store_service_instance.")
    except Exception as e_vs:
        logger.error(f"An error occurred during vector store service initialization: {e_vs}")


    logger.info("FastAPI application startup complete.")
    
    yield  # Application runs after this point

    # Actions to perform on application shutdown
    logger.info("FastAPI application shutting down...")
    # If specific close methods were available for services, they'd be called here.
    # e.g., await some_service.close()
    logger.info("FastAPI application shutdown complete.")

# Initialise FastAPI app with the lifespan manager
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan  # Register the lifespan context manager
)

# --- CORS Middleware (if needed) ---
origins = [
    "http://localhost",          # Streamlit default in dev
    "http://localhost:8501",     # Streamlit default explicit port
    # Add your frontend production URL if deployed
    "http://127.0.0.1",          # Add this if accessing frontend via 127.0.0.1
    "http://127.0.0.1:8501"      # Add this if accessing frontend via 127.0.0.1:8501
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Include API Routers ---
app.include_router(api_endpoints.router, prefix="/api/v1", tags=["RAG Operations"])


# --- Root and Health Endpoints ---
@app.get("/", tags=["Root"])
async def read_root() -> dict:
    """
    Root endpoint providing a welcome message.
    """
    return {"message": f"Welcome to the {API_TITLE}!"}

@app.get("/health", response_class=JSONResponse, status_code=200)
async def health_check():
    """
    Simple health check endpoint.
    Returns HTTP 200 with return {"status": "healthy"} if the server is up.
    """
    return {"status": "healthy"}

# To run (from the `rag_application_assistant` directory):
# cd backend
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
