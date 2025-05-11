# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # If frontend is on a different domain/port

from backend.app.api import endpoints as api_endpoints
from backend.app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, get_logger, GENERATIVE_MODEL_NAME

# from backend.app.services.vector_store_service import vector_store_service_instance # Ensure it's initialized
# from backend.app.services.document_processor_service import document_processor_instance

logger = get_logger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# --- CORS Middleware (if needed) ---
# origins = [
#     "http://localhost",          # Streamlit default in dev
#     "http://localhost:8501",     # Streamlit default explicit port
#     # Add your frontend production URL if deployed
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# --- Include API Routers ---
app.include_router(api_endpoints.router, prefix="/api/v1", tags=["RAG Operations"])


@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    e.g., initializing database connections, loading models.
    Services are now mostly initialized at import time or via singleton pattern.
    This is a good place to log startup messages or perform final checks.
    """
    logger.info("FastAPI application startup commencing...")
    # Ensure vector store is ready (it initializes itself as a singleton)
    # from backend.app.services.vector_store_service import vector_store_service_instance
    # if vector_store_service_instance.client and vector_store_service_instance.collection:
    #     logger.info(f"Vector store service connected. Collection '{vector_store_service_instance.collection_name}' count: {vector_store_service_instance.get_collection_count()}")
    # else:
    #     logger.error("Vector store service failed to initialize properly during startup.")

    # Ensure LLM service is ready
    from backend.app.services.llm_service import generative_model
    if generative_model:
        logger.info(f"LLM service with model '{GENERATIVE_MODEL_NAME}' is ready.")
    else:
        logger.error("LLM service failed to initialize generative model (check GOOGLE_API_KEY).")

    logger.info("FastAPI application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown.
    e.g., closing database connections.
    """
    logger.info("FastAPI application shutting down...")
    # ChromaDB client might have its own cleanup, but typically managed by Python's GC
    # If specific close methods were available for services, they'd be called here.
    logger.info("FastAPI application shutdown complete.")


@app.get("/", tags=["Root"])
async def read_root() -> dict:
    """
    Root endpoint providing a welcome message.
    """
    return {"message": f"Welcome to the {API_TITLE}!"}

# To run (from the `rag_application_assistant` directory):
# cd backend
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000