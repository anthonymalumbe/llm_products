# backend/app/api/endpoints.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Body
from typing import List, Dict

from app.schemas.models import (
    ChatMessage, ValidationSummaryResponse, GuidelineProcessingResponse,
    ChatResponse, ErrorDetail
)
from app.services.document_processor_service import DocumentProcessorService, document_processor_instance
from app.core.config import get_logger

# from backend.app.schemas.models import (
#     ChatMessage, ValidationSummaryResponse, GuidelineProcessingResponse,
#     ChatResponse, ErrorDetail
# )
# from backend.app.services.document_processor_service import DocumentProcessorService, document_processor_instance
# from backend.app.core.config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Dependency for DocumentProcessorService (can be more sophisticated with FastAPI's Depends)
def get_doc_processor_service() -> DocumentProcessorService:
    # In a more complex app, this could fetch a request-scoped or singleton instance
    return document_processor_instance


@router.post(
    "/upload_guidelines/",
    response_model=GuidelineProcessingResponse,
    summary="Upload and Index Guideline PDFs",
    responses={
        200: {"description": "Guidelines processed successfully."},
        400: {"model": ErrorDetail, "description": "Bad request (e.g., no files, wrong file type)."},
        500: {"model": ErrorDetail, "description": "Internal server error during processing."},
        503: {"model": ErrorDetail, "description": "Service unavailable (e.g., vector store not ready)."}
    }
)
async def upload_guidelines_endpoint(
    files: List[UploadFile] = File(..., description="List of PDF files to be processed as guidelines."),
    doc_processor: DocumentProcessorService = Depends(get_doc_processor_service)
):
    """
    Receives guideline PDF files, processes them, and indexes their content.
    Existing guidelines are cleared before new ones are added.
    """
    if not files:
        logger.warning("No files received for guideline upload endpoint.")
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    for file_upload in files:
        if not (file_upload.filename and file_upload.filename.lower().endswith('.pdf')):
            logger.warning(f"Rejected non-PDF file during guideline upload: {file_upload.filename}")
            # Close all files if one is bad
            for f in files: f.file.close()
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted for guidelines. Received: {file_upload.filename or 'Unknown filetype'}"
            )
    
    logger.info(f"Received {len(files)} guideline file(s) for processing via API.")
    try:
        message = doc_processor.process_and_index_guidelines(files)
        return GuidelineProcessingResponse(message=message)
    except RuntimeError as e: # Catch specific errors from the service
        logger.error(f"Runtime error during guideline processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process guidelines: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing guideline upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        # Ensure all uploaded files are closed, even on error
        for file_upload in files:
            if hasattr(file_upload.file, 'closed') and not file_upload.file.closed:
                file_upload.file.close()


@router.post(
    "/validate_application/",
    response_model=ValidationSummaryResponse,
    summary="Validate Application PDF",
     responses={
        200: {"description": "Validation successful, summary returned."},
        400: {"model": ErrorDetail, "description": "Bad request (e.g., wrong file type)."},
        422: {"model": ErrorDetail, "description": "Unprocessable entity (e.g., PDF content error, LLM extraction failed)."},
        500: {"model": ErrorDetail, "description": "Internal server error during validation."},
        503: {"model": ErrorDetail, "description": "Service unavailable."}
    }
)
async def validate_application_api_endpoint( # Renamed from endpoint to avoid clash
    file: UploadFile = File(..., description="A single PDF file representing the application form."),
    doc_processor: DocumentProcessorService = Depends(get_doc_processor_service)
):
    """
    Uploads an application PDF, extracts data, validates against guidelines,
    and returns a formatted summary.
    """
    logger.info(f"Received application file: {file.filename} for validation via API.")
    if not (file.filename and file.filename.lower().endswith('.pdf')):
        logger.warning(f"Rejected non-PDF application file: {file.filename}")
        file.file.close()
        raise HTTPException(status_code=400, detail=f"Only PDF files are accepted. Received: {file.filename or 'Unknown filetype'}")

    try:
        formatted_summary = doc_processor.validate_application_pdf(file) # file object is passed, service handles temp saving
        
        if formatted_summary.startswith("Error:"):
             logger.error(f"Validation process for {file.filename} returned an error: {formatted_summary}")
             raise HTTPException(status_code=422, detail=formatted_summary) # Unprocessable Entity

        return ValidationSummaryResponse(validation_summary=formatted_summary)
    except RuntimeError as e:
        logger.error(f"Runtime error during application validation for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to validate application: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error validating application {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    # finally: # File is closed within the service method now
        # if hasattr(file.file, 'closed') and not file.file.closed:
        #     file.file.close()


@router.post(
    "/chat_messages/",
    response_model=ChatResponse,
    summary="Handle Chat Message",
    responses={
        200: {"description": "Chat response successful."},
        500: {"model": ErrorDetail, "description": "Internal server error during chat processing."},
        503: {"model": ErrorDetail, "description": "Chat service unavailable."}
    }
)
async def handle_chat_message_endpoint(
    chat_message: ChatMessage = Body(...),
    doc_processor: DocumentProcessorService = Depends(get_doc_processor_service)
):
    """
    Receives a chat message and returns a RAG-based response.
    """
    logger.info(f"Received chat message via API: '{chat_message.message[:100]}...'")
    try:
        response_text = doc_processor.get_chat_response(chat_message.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@router.get(
    "/health",
    summary="Health Check",
    response_model=Dict[str, str]
)
async def health_check():
    """Simple health check endpoint."""
    # Could add checks for DB connection, LLM service availability here
    return {"status": "healthy", "message": "API is running."}