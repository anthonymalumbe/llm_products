# backend/app/services/document_processor_service.py
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

# from backend.app.core.config import get_logger
# from backend.app.utils.file_processing import extract_text_from_pdf, chunk_text
# from backend.app.services.vector_store_service import VectorStoreService, vector_store_service_instance
# from backend.app.services.llm_service import (
#     generate_dictionary_from_text,
#     validate_entry_with_llm,
#     format_validation_results_with_llm,
#     generate_chat_response
# )

from app.core.config import get_logger
from app.utils.file_processing import extract_text_from_pdf, chunk_text
from app.services.vector_store_service import VectorStoreService, vector_store_service_instance
from app.services.llm_service import (
    generate_dictionary_from_text,
    validate_entry_with_llm,
    format_validation_results_with_llm,
    generate_chat_response
)

logger = get_logger(__name__)

class DocumentProcessorService:
    """
    Orchestrates document processing, indexing, validation, and chat functionalities.
    """
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        logger.info("DocumentProcessorService initialized.")

    def process_and_index_guidelines(self, pdf_files: List[UploadFile]) -> str:
        """
        Processes uploaded guideline PDFs: extracts text, chunks, and indexes them.
        Clears the existing collection before indexing.

        Args:
            pdf_files (List[UploadFile]): List of guideline PDF files.

        Returns:
            str: A message indicating the result of the operation.
        
        Raises:
            RuntimeError: If critical operations like clearing collection fail.
        """
        if not pdf_files:
            logger.warning("No PDF files provided for guideline processing.")
            return "No guideline PDF files provided."

        if not self.vector_store.client or not self.vector_store.embed_fn:
             logger.error("Vector store client or embed_fn not initialized. Cannot process guidelines.")
             raise RuntimeError("Guideline processing system is not properly initialized.")

        # 1. Clear existing collection
        logger.info("Clearing existing guideline collection before indexing new documents.")
        if not self.vector_store.clear_collection():
            logger.error("Failed to clear and recreate guideline collection. Aborting indexing.")
            raise RuntimeError("Failed to prepare guideline database for new documents.")
        
        all_chunks_for_db: List[str] = []
        all_metadatas_for_db: List[Dict[str, Any]] = []
        all_ids_for_db: List[str] = []
        doc_id_counter = 0
        processed_filenames: List[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for pdf_file in pdf_files:
                if not pdf_file.filename:
                    logger.warning("Received a file without a filename, skipping.")
                    continue
                
                processed_filenames.append(pdf_file.filename)
                temp_file_path = os.path.join(tmpdir, pdf_file.filename)
                
                try:
                    with open(temp_file_path, "wb") as f:
                        shutil.copyfileobj(pdf_file.file, f)
                    logger.info(f"Temporarily saved guideline file: {temp_file_path}")

                    pdf_text = extract_text_from_pdf(temp_file_path)
                    if not pdf_text:
                        logger.warning(f"No text extracted from {pdf_file.filename}, skipping.")
                        continue

                    chunks = chunk_text(pdf_text)
                    if not chunks:
                        logger.warning(f"No chunks created for {pdf_file.filename}, skipping.")
                        continue
                    
                    logger.info(f"Generated {len(chunks)} chunks for {pdf_file.filename}.")
                    for i, chunk_content in enumerate(chunks):
                        chunk_id = f"{pdf_file.filename}_chunk_{doc_id_counter}"
                        all_chunks_for_db.append(chunk_content)
                        all_metadatas_for_db.append({"source_pdf": pdf_file.filename, "chunk_index": i})
                        all_ids_for_db.append(chunk_id)
                        doc_id_counter += 1
                
                except Exception as e:
                    logger.error(f"Error processing guideline file {pdf_file.filename}: {e}", exc_info=True)
                    # Continue to next file
                finally:
                    pdf_file.file.close() # Ensure file is closed

        if not all_chunks_for_db:
            logger.warning("No text chunks were generated from any of the provided PDFs. Indexing cannot proceed.")
            return f"No text content suitable for indexing found in {', '.join(processed_filenames) if processed_filenames else 'uploaded files'}."

        # 2. Add new documents to the (now empty) collection
        logger.info(f"Indexing {len(all_chunks_for_db)} total chunks from {len(processed_filenames)} PDFs.")
        if self.vector_store.add_documents(all_chunks_for_db, all_metadatas_for_db, all_ids_for_db):
            msg = f"Successfully processed and indexed {len(all_chunks_for_db)} chunks from {len(processed_filenames)} guideline file(s): {', '.join(processed_filenames)}."
            logger.info(msg)
            return msg
        else:
            logger.error("Failed to add documents to the vector store during guideline processing.")
            raise RuntimeError("Failed to index guideline documents into the vector store.")


    def validate_application_pdf(self, application_pdf: UploadFile) -> str:
        """
        Processes an uploaded application PDF, extracts data, validates it against
        indexed guidelines, and formats the results.

        Args:
            application_pdf (UploadFile): The application PDF file.

        Returns:
            str: A formatted string summarizing validation results or an error message.
        
        Raises:
            RuntimeError: If a critical system component is unavailable.
        """
        if not application_pdf.filename:
            return "Error: Application PDF has no filename."

        if self.vector_store.get_collection_count() == 0:
             logger.warning("Attempting to validate application with an empty guideline collection. Results will be uncertain.")
             # Proceed, but validation will reflect no guidelines.

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file_path = os.path.join(tmpdir, application_pdf.filename)
            try:
                with open(temp_file_path, "wb") as f:
                    shutil.copyfileobj(application_pdf.file, f)
                logger.info(f"Temporarily saved application file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error saving temporary application file {application_pdf.filename}: {e}", exc_info=True)
                return f"Error: Could not save uploaded application file '{application_pdf.filename}'."
            finally:
                application_pdf.file.close()

            # 1. Extract text from application PDF
            application_text = extract_text_from_pdf(temp_file_path)
            if not application_text:
                return f"Error: Could not extract text from application PDF '{application_pdf.filename}'."

            # 2. Extract structured data (dictionary) from application text using LLM
            logger.info(f"Extracting structured data from application '{application_pdf.filename}' using LLM.")
            application_data = generate_dictionary_from_text(application_text)
            if not application_data:
                return f"Error: Could not extract structured data (form fields) from '{application_pdf.filename}' using LLM. The PDF might be image-based or have an unsupported format."
            
            logger.info(f"Extracted {len(application_data)} fields from '{application_pdf.filename}'. Starting validation.")

            # 3. Validate each field against guidelines
            validation_results: List[Dict[str, Any]] = []
            for field_name, field_value in application_data.items():
                query = f"What are the guidelines, rules, or requirements for the application form field named: '{field_name}' with a submitted value of '{str(field_value)[:50]}'?"
                
                relevant_guidelines = self.vector_store.query_collection(query_text=query, n_results=5)
                
                # if not relevant_guidelines:
                #     logger.info(f"No specific guidelines found for field '{field_name}' via RAG.")
                # else:
                #     logger.info(f"Found {len(relevant_guidelines)} guideline snippets for field '{field_name}'.")

                result_item = validate_entry_with_llm(field_name, field_value, relevant_guidelines)
                validation_results.append(result_item)
            
            logger.info(f"Completed validation for {len(validation_results)} fields from '{application_pdf.filename}'.")

            # 4. Format validation results using LLM
            formatted_summary = format_validation_results_with_llm(validation_results)
            logger.info(f"Formatted validation summary for '{application_pdf.filename}'.")
            return formatted_summary

    def get_chat_response(self, user_query: str) -> str:
        """
        Handles a user chat query by retrieving relevant guidelines and generating an LLM response.
        """
        if self.vector_store.get_collection_count() == 0:
            logger.warning("Chat query received, but guideline collection is empty.")
            # Let generate_chat_response handle this, it has a specific prompt for no context
        
        # 1. Retrieve relevant guidelines
        # A more generic query for chat might be just the user_query itself,
        # or a slight modification to target guideline retrieval.
        context_query = f"Information related to: {user_query}"
        relevant_guidelines = self.vector_store.query_collection(query_text=context_query, n_results=5)

        # 2. Generate response using LLM with context
        response = generate_chat_response(user_query, relevant_guidelines)
        return response

# Global instance or use FastAPI dependency injection
document_processor_instance = DocumentProcessorService(vector_store=vector_store_service_instance)