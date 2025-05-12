# Building an AI-Powered Application Validation Assistant with RAG

Processing application forms against complex guidelines can be a tedious and error-prone task. What if we could automate parts of this process using AI? This post walks through an application built to do just that: an "Application Validation Assistant" that uses Retrieval-Augmented Generation (RAG) to help users validate application forms against uploaded guideline documents and ask questions about those guidelines.

We'll dive deeper into the two main components: a user-friendly frontend built with Streamlit and a robust backend powered by FastAPI, exploring the specific implementation details.

## What Does the Application Do?

The core purpose of this application is to streamline the validation of application PDFs against a set of guideline PDFs. It offers three main functionalities, handled by specific backend services:

1.  **Guideline Processing:** Users upload guideline PDFs via the Streamlit UI. The backend receives these, extracts text, chunks it, generates embeddings using a Google Gemini model (`models/text-embedding-004` by default), and indexes these embeddings and the text chunks into a ChromaDB vector store. The existing collection is cleared before adding new guidelines.
2.  **Application Validation:** Users upload an application PDF. The backend extracts text from the PDF. It then uses a generative LLM (like `gemini-1.5-flash`) to parse the text into key-value pairs representing form fields. For each field, it queries the ChromaDB vector store for relevant guideline snippets. The field's value and the retrieved context are sent to the LLM to determine if the entry is valid, invalid, or uncertain, providing a reason. Finally, the LLM formats all results into a human-readable summary.
3.  **Chat with Guidelines:** Users ask natural language questions via the Streamlit chat interface. The backend queries the ChromaDB vector store for context relevant to the question and then uses the LLM, along with the retrieved context, to generate an informative answer.

## The Frontend: Streamlit for Interactive UI (`app.py`)

The user interface is built using Streamlit, a popular Python library for creating interactive web apps for data science and machine learning projects. Key elements include:

* **Layout:** Wide layout with sections for guideline processing, application validation, results display, and chat.
    ```python
    # frontend/app.py
    import streamlit as st
    import pandas as pd
    import requests
    # ... other imports

    st.set_page_config(page_title="RAG Application Assistant", layout="wide")
    st.title("📑 Application Validation Assistant")
    st.markdown("""
    Welcome! This tool helps you work with application guidelines.
    1.  **Upload Guideline PDFs**: Index the content of your guideline documents.
    2.  **Upload Application PDF**: Submit an application form for automated validation against the guidelines.
    3.  **Chat with Guidelines**: Ask questions about the processed guideline documents.
    """)

    # --- Initialize Session State Variables ---
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # ... other session state initializations
    ```
* **File Uploads:** Uses `st.file_uploader` for PDF uploads and buttons to trigger backend API calls.
    ```python
    # frontend/app.py - Snippet from Guideline Upload Section
    with col1: # Assuming col1 is a Streamlit column
        st.subheader("1. Process Guidelines")
        uploaded_guideline_files = st.file_uploader(
            "Upload Guideline PDF(s)", type="pdf", accept_multiple_files=True,
            key="guideline_uploader", # ...
        )

        if uploaded_guideline_files:
            if st.button("Process Selected Guidelines", # ...):
                files_to_send = [('files', (file.name, file.getvalue(), 'application/pdf'))
                                 for file in uploaded_guideline_files]
                with st.spinner("Processing guidelines..."):
                    try:
                        # Send files to backend API endpoint
                        response = requests.post(f"{BACKEND_API_BASE_URL}/upload_guidelines/",
                                                 files=files_to_send, timeout=120)
                        # ... handle response ...
                    except requests.exceptions.RequestException as e:
                        # ... handle error ...
    ```
* **Validation Display:** Parses the LLM-formatted summary from the backend (using regex in `display_validation_results`) and displays it using Pandas DataFrames and Streamlit metrics, styled with custom CSS.
    ```python
    # frontend/app.py - Snippet from display_validation_results
    def display_validation_results(raw_summary: Optional[str]):
        # ... (parsing logic using regex to find fields, status, reason) ...

        if results_data: # list of dicts parsed from raw_summary
            df = pd.DataFrame(results_data)
            if not df.empty:
                df.insert(0, 'No.', range(1, 1 + len(df)))

            # Apply CSS styling
            st.markdown("""<style> .dataframe table { ... } </style>""", unsafe_allow_html=True)
            # Display table
            st.markdown(df[["No.", "Field", "Value", "Status", "Reason"]]
                        .to_html(escape=False, index=False, classes="dataframe"),
                        unsafe_allow_html=True)

        # Display summary metrics if available (parsed via regex)
        if summary_section_match:
             passed_count = int(summary_section_match.group(1))
             failed_count = int(summary_section_match.group(2))
             uncertain_count = int(summary_section_match.group(3))
             # ... use st.metric to show passed, failed, uncertain counts ...
    ```
* **Chat Interface:** Uses `st.chat_input` and `st.chat_message` for the chat experience, sending messages to the backend `/chat_messages/` endpoint.
    ```python
    # frontend/app.py - Snippet from Chat Section
    st.header("💬 Chat with Guidelines")
    # ... (display previous messages using st.chat_message) ...

    if prompt := st.chat_input("Your question...", # ...):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"): st.markdown(prompt) # Simplified

        with st.spinner("Thinking..."):
            try:
                # Send message to backend chat endpoint
                response = requests.post(f"{BACKEND_API_BASE_URL}/chat_messages/",
                                         json={"message": prompt}, timeout=60)
                response.raise_for_status()
                assistant_response = response.json().get("response", "Error")
            except requests.exceptions.RequestException as e:
                # ... handle error ...
                assistant_response = f"Chat Error: {e}" # Simplified

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="🤖"): st.markdown(assistant_response) # Simplified
    ```
* **Backend Communication:** Uses `requests` to interact with FastAPI endpoints. Includes a basic health check on startup.
* **Technology Showcase:** The UI includes a "Powered By" section displaying logos of the key technologies used (Streamlit, FastAPI, Gemini, LangChain, ChromaDB), loaded dynamically using base64 encoding.

## The Backend: FastAPI & Core Services

The backend uses FastAPI to provide the API and orchestrates several services to handle the core logic. Configuration values like API keys, model names, and database paths are managed via `config.py` and loaded from environment variables.

* **API Endpoints (`endpoints.py`):** Defines the routes using FastAPI's `APIRouter`. It handles request validation (e.g., checking for PDF files) and calls the appropriate service methods. Dependency Injection (`Depends`) is used to provide the `DocumentProcessorService` instance to the endpoint functions.
    ```python
    # backend/app/api/endpoints.py
    from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Body
    from typing import List
    # Other necessary imports (schemas, services)
    from app.schemas.models import (
        ChatMessage, ValidationSummaryResponse, GuidelineProcessingResponse,
        ChatResponse, ErrorDetail # Pydantic models
    )
    from app.services.document_processor_service import DocumentProcessorService, document_processor_instance

    router = APIRouter()

    # Dependency function
    def get_doc_processor_service() -> DocumentProcessorService:
        return document_processor_instance

    @router.post(
        "/upload_guidelines/",
        response_model=GuidelineProcessingResponse,
        summary="Upload and Index Guideline PDFs",
        # ... responses definition ...
    )
    async def upload_guidelines_endpoint(
        files: List[UploadFile] = File(..., description="List of PDF files..."),
        doc_processor: DocumentProcessorService = Depends(get_doc_processor_service)
    ):
        # --- File validation logic (check if PDF, etc.) ---
        if not files:
             raise HTTPException(status_code=400, detail="No files were uploaded.")
        # (Add loop to check file types and raise 400 if non-PDF)

        try:
            message = doc_processor.process_and_index_guidelines(files) # Call service
            return GuidelineProcessingResponse(message=message)
        except RuntimeError as e:
             raise HTTPException(status_code=500, detail=f"Failed to process guidelines: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
        finally:
             # Ensure files are closed
             for f in files: f.file.close()


    @router.post("/validate_application/", response_model=ValidationSummaryResponse, # ...)
    async def validate_application_api_endpoint(
        file: UploadFile = File(..., description="A single PDF file..."),
        doc_processor: DocumentProcessorService = Depends(get_doc_processor_service)
    ):
        # --- File validation logic (check if PDF) ---
        if not (file.filename and file.filename.lower().endswith('.pdf')):
             raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        try:
            formatted_summary = doc_processor.validate_application_pdf(file) # Call service
            if formatted_summary.startswith("Error:"): # Check if service indicated an error
                 raise HTTPException(status_code=422, detail=formatted_summary)
            return ValidationSummaryResponse(validation_summary=formatted_summary)
        except RuntimeError as e:
             raise HTTPException(status_code=500, detail=f"Failed to validate application: {str(e)}")
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
        # File closing is handled within the service in this implementation


    @router.post("/chat_messages/", response_model=ChatResponse, # ...)
    async def handle_chat_message_endpoint(
        chat_message: ChatMessage = Body(...), # Use Pydantic model for request body
        doc_processor: DocumentProcessorService = Depends(get_doc_processor_service)
    ):
        try:
            response_text = doc_processor.get_chat_response(chat_message.message) # Call service
            return ChatResponse(response=response_text) # Use Pydantic model for response
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

    ```
* **File Processing (`file_processing.py`):** Handles the low-level PDF interaction and text splitting.
    * `extract_text_from_pdf`: Uses `pypdf` library to read PDF files page by page and extract text content.
    * `chunk_text`: Implements a two-pass chunking strategy using LangChain's `RecursiveCharacterTextSplitter` and `SentenceTransformersTokenTextSplitter`. Chunk size and overlap parameters are configurable via `config.py`.
        ```python
        # backend/app/utils/file_processing.py
        from pypdf import PdfReader
        from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
        from app.core.config import CHARACTER_CHUNK_SIZE, TOKEN_CHUNK_SIZE # Import config values

        def extract_text_from_pdf(pdf_path: str) -> str:
            # ... implementation using PdfReader ...
            try:
                reader = PdfReader(pdf_path)
                pdf_text = ""
                for i, page in enumerate(reader.pages):
                     text = page.extract_text()
                     if text: pdf_text += f"\n--- Page {i+1} ---\n{text.strip()}"
                return pdf_text
            except Exception as e:
                # logger.error(...)
                return ""

        def chunk_text(text: str, # ... chunk size params from config) -> List[str]:
            if not text: return []
            # First pass: character‐aware splitting
            char_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=CHARACTER_CHUNK_SIZE, # From config
                chunk_overlap=CHARACTER_CHUNK_OVERLAP, # From config
                # ...
            )
            char_chunks = char_splitter.split_text(text)

            # Second pass: token‐aware splitting
            token_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=TOKEN_CHUNK_OVERLAP, # From config
                tokens_per_chunk=TOKEN_CHUNK_SIZE, # From config
            )
            all_token_chunks = []
            for char_chunk in char_chunks:
                 if not char_chunk.strip(): continue
                 try:
                      token_chunks_from_char_chunk = token_splitter.split_text(char_chunk)
                      all_token_chunks.extend(token_chunks_from_char_chunk)
                 except Exception as e:
                      # logger.error(...)
                      pass # Optionally handle/log error
            # ... clean and return final_chunks ...
            cleaned_chunks = [re.sub(r'\s+', ' ', c).strip() for c in all_token_chunks]
            return [c for c in cleaned_chunks if c]
        ```
* **Vector Store (`vector_store_service.py`):** Manages ChromaDB interactions.
    * Initializes a `PersistentClient` pointing to the path specified in `config.py`.
    * Uses `get_or_create_collection` to load or set up the guideline collection, crucially passing the custom `GeminiEmbeddingFunction`.
    * `clear_collection`: Deletes and recreates the collection.
    * `add_documents`: Adds text chunks, metadata, and IDs in batches, setting the embedding task type to `"retrieval_document"`.
    * `query_collection`: Queries the collection using text, setting the embedding task type to `"retrieval_query"`, and retrieves the document text (`include=['documents']`).
        ```python
        # backend/app/services/vector_store_service.py
        import chromadb
        from app.core.config import CHROMA_DB_PATH, COLLECTION_NAME # Import config values
        from app.services.llm_service import GeminiEmbeddingFunction # Import custom embedding function

        class VectorStoreService:
            # ... singleton pattern implementation ...
            def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
                # ... prevent re-initialization ...
                self.db_path = db_path
                self.collection_name = collection_name
                self.client: Optional[chromadb.PersistentClient] = None
                self.collection: Optional[chromadb.Collection] = None
                # Instantiate our specific embedding function
                self.embed_fn = GeminiEmbeddingFunction()
                try:
                     self.client = chromadb.PersistentClient(path=self.db_path)
                     self._load_or_create_collection()
                     # logger.info("VectorStoreService initialized successfully.")
                except Exception as e:
                     # logger.error(...)
                     self.client = None
                     self.collection = None
                self._initialized = True

            def _load_or_create_collection(self):
                 # logger.info(f"Getting or creating collection: {self.collection_name}")
                 self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    # Pass the instance of our custom embedding function
                    embedding_function=self.embed_fn
                )
                 # logger.info(f"Collection '{self.collection_name}' ready.")

            def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
                if not self.collection: return False
                # Ensure the embedding function is set for indexing task
                self.embed_fn.set_task_type("retrieval_document")
                batch_size = 100 # Example batch size
                # logger.info(f"Adding {len(chunks)} documents to collection...")
                try:
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i+batch_size]
                        batch_metadatas = metadatas[i:i+batch_size]
                        batch_ids = ids[i:i+batch_size]
                        if not batch_chunks: continue
                        self.collection.add(
                            documents=batch_chunks,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                    # logger.info(f"Successfully added {len(chunks)} documents.")
                    return True
                except Exception as e:
                    # logger.error(...)
                    return False

            def query_collection(self, query_text: str, n_results: int = 5) -> List[str]:
                if not self.collection: return []
                # Set the embedding function task type specifically for querying
                self.embed_fn.set_task_type("retrieval_query")
                try:
                    results = self.collection.query(
                        query_texts=[query_text],
                        n_results=n_results,
                        # Ensure we retrieve the document text content
                        include=['documents']
                    )
                    # Extract the document texts from the results structure
                    retrieved_docs = results['documents'][0] if results and results.get('documents') and results['documents'][0] else []
                    return retrieved_docs
                except Exception as e:
                    # logger.error(...)
                    return []

            def get_collection_count(self) -> int:
                # ... implementation to return self.collection.count() ...
                pass

        # Global instance
        vector_store_service_instance = VectorStoreService()
        ```
* **LLM Service (`llm_service.py`):** Handles all direct interactions with the Google Gemini models.
    * Initializes the generative model (`gemini-1.5-flash` or as configured) using the API key.
    * `GeminiEmbeddingFunction`: Custom class interfacing with ChromaDB's `EmbeddingFunction`, using `genai.embed_content` with retry logic and dynamic `task_type`.
    * `generate_dictionary_from_text`: Prompts the LLM to extract key-value pairs from application text and return a Python dictionary string (parsed via `ast.literal_eval`).
    * `validate_entry_with_llm`: Constructs a detailed prompt including field details and retrieved context, instructing the LLM to output `VALIDATION_STATUS` and `EXPLANATION` (parsed via regex).
    * `format_validation_results_with_llm`: Sends individual validation results (as JSON) to the LLM with instructions to format a human-readable summary.
    * `generate_chat_response`: Builds a prompt with the user query and retrieved context (if any), instructing the LLM to answer based *only* on the provided context.
        ```python
        # backend/app/services/llm_service.py
        import google.generativeai as genai
        from google.generativeai import types as genai_types
        from google.api_core import retry
        from chromadb import EmbeddingFunction, Documents, Embeddings
        from app.core.config import EMBEDDING_MODEL_NAME, GENERATIVE_MODEL_NAME, GOOGLE_API_KEY # Import config
        import ast, re, json, textwrap

        # --- Initialize Generative Model ---
        generative_model = None
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                generative_model = genai.GenerativeModel(GENERATIVE_MODEL_NAME) # Use config
            except Exception as e:
                # logger.error(...)
                pass
        # ...

        # --- Custom Embedding Function for ChromaDB ---
        class GeminiEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, task_type: str = "retrieval_document"): # Use config
                self._model_name = model_name
                self._task_type = task_type
                # ...

            # Simplified embedding call (actual uses retry decorator)
            def _embed_with_retry(self, texts: List[str]) -> Optional[List[List[float]]]:
                 try:
                     response = genai.embed_content(
                         model=self._model_name,
                         content=texts,
                         task_type=self._task_type # Use dynamic task type
                     )
                     return response.get('embedding')
                 except Exception as e:
                      # logger.error(...)
                      return None

            def __call__(self, input_texts: Documents) -> Embeddings:
                 embeddings = self._embed_with_retry(texts=input_texts)
                 if embeddings is None: return [[] for _ in input_texts] # Handle failure
                 return embeddings

            def set_task_type(self, task_type: str): # Allow changing task type
                valid_task_types = ["retrieval_document", "retrieval_query", ...] # Add other valid types
                if task_type not in valid_task_types:
                    raise ValueError(f"Invalid task_type: {task_type}")
                self._task_type = task_type
                # ...

        # --- LLM Calls ---
        def generate_dictionary_from_text(text: str) -> Optional[Dict[str, Any]]:
            if not generative_model: return None
            prompt = f"""Extract all input fields... Return ONLY a valid Python dictionary... Form text:\n---\n{text}\n---\nExtracted Dictionary:""" # Simplified
            try:
                response = generative_model.generate_content(prompt, generation_config=genai_types.GenerationConfig(temperature=0.0))
                output_text = response.text.strip()
                match = re.search(r"(\{[\s\S]*?\})", output_text) # Look for dict in output
                dict_str = match.group(1) if match else output_text
                return ast.literal_eval(dict_str)
            except Exception as e:
                # logger.error(...)
                return None

        def _build_validation_prompt(field_name: str, field_value: Any, guidelines: List[str]) -> str:
             guideline_context = "\\n\\n---\\n\\n".join(guidelines) if guidelines else "No specific guidelines provided..."
             # Construct the detailed prompt instructing LLM to output VALIDATION_STATUS and EXPLANATION
             return f"""Analyze... Field: "{field_name}" Value: "{field_value}" Context:\n{guideline_context}\n...Instructions:\n... Respond *only* with:\nVALIDATION_STATUS: [VALID/INVALID/CANNOT DETERMINE]\nEXPLANATION: [...]""" # Simplified

        def validate_entry_with_llm(field_name: str, field_value: Any, guidelines: List[str]) -> Dict[str, Any]:
            if not generative_model: return {"isValid": None, "reason": "LLM unavailable"} # Simplified error
            prompt = _build_validation_prompt(field_name, field_value, guidelines)
            if not guidelines: # Handle case where RAG found nothing
                 return {"field": field_name, "value": field_value, "isValid": None, "reason": "No relevant guidelines found."}
            try:
                response = generative_model.generate_content(prompt, generation_config=genai_types.GenerationConfig(temperature=0.1))
                raw_text = response.text.strip()
                # Parse raw_text using regex for status and explanation
                status_match = re.search(r"VALIDATION_STATUS\s*:\s*(VALID|INVALID|CANNOT\s*DETERMINE)", raw_text, re.IGNORECASE)
                explanation_match = re.search(r"EXPLANATION\s*:\s*([\s\S]+)", raw_text, re.IGNORECASE)
                # ... extract status and explanation, map to True/False/None ...
                is_valid = ... # Mapped value
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation."
                return {"field": field_name, "value": field_value, "isValid": is_valid, "reason": explanation}
            except Exception as e:
                # logger.error(...)
                return {"field": field_name, "value": field_value, "isValid": None, "reason": f"LLM Error: {e}"}


        def format_validation_results_with_llm(validation_results: List[Dict[str, Any]]) -> str:
            if not generative_model: return "LLM Formatter unavailable"
            if not validation_results: return "Nothing to format."
            results_json_str = json.dumps(validation_results, indent=2)
            # Construct prompt asking LLM to format the JSON into the desired text summary with icons
            prompt = f"""Given the JSON results... produce a concise summary... Use icons (✅, ❌, ❓)... Use '---' separator... Conclude with 'Summary' section...\nJSON Results:\n```json\n{results_json_str}\n```\nDesired Output Format:...""" # Simplified
            try:
                 response = generative_model.generate_content(prompt, generation_config=genai_types.GenerationConfig(temperature=0.1))
                 return response.text.strip()
            except Exception as e:
                 # logger.error(...)
                 # Provide basic fallback formatting
                 return "LLM Formatting Failed.\n" + json.dumps(validation_results, indent=2) # Basic fallback


        def generate_chat_response(user_query: str, context_guidelines: List[str]) -> str:
            if not generative_model: return "Chat LLM unavailable"
            # Build prompt with or without context
            if not context_guidelines:
                prompt = f"""User asks: "{user_query}". No specific info found. Respond politely stating info unavailable or answer generally."""
            else:
                guideline_context_str = "\\n\\n---\\n\\n".join(context_guidelines)
                prompt = f"""Answer based ONLY on "Guideline Context". If irrelevant, state info not found. Be concise. Context:\n--- START ---\n{guideline_context_str}\n--- END ---\nUser Query: "{user_query}"\nAnswer:""" # Simplified
            try:
                response = generative_model.generate_content(prompt, generation_config=genai_types.GenerationConfig(temperature=0.2))
                return response.text.strip()
            except Exception as e:
                # logger.error(...)
                return f"Chat Error: {e}"
        ```
* **Document Processing Service (`document_processor_service.py`):** Orchestrates the workflow, tying the other services together.
    * `process_and_index_guidelines`: Coordinates file processing, vector store clearing, and adding documents.
    * `validate_application_pdf`: Coordinates PDF extraction, LLM data extraction, vector store querying, LLM validation per field, and final LLM formatting.
    * `get_chat_response`: Coordinates vector store querying and LLM response generation.

## The RAG Architecture In Action: Theory and Implementation

This application leverages a **Retrieval-Augmented Generation (RAG)** architecture. RAG is a powerful AI framework designed to improve the quality, accuracy, and relevance of Large Language Model (LLM) responses. Instead of relying solely on the vast but potentially outdated or generic knowledge the LLM was trained on, RAG connects the LLM to external, authoritative knowledge sources in real-time.

**Why RAG?**

* **Reduces Hallucinations:** By grounding the LLM's response in specific, retrieved facts, RAG minimizes the chances of the model generating incorrect or nonsensical information ("hallucinations").
* **Uses Current & Specific Data:** It allows the LLM to access and utilize up-to-date information or domain-specific knowledge (like our application guidelines) that wasn't part of its original training data.
* **Increases Trust & Transparency:** RAG enables models to potentially cite their sources (though not explicitly implemented here, it's a key benefit), allowing users to verify the information.
* **Cost-Effective:** Compared to the high cost and complexity of retraining or fine-tuning an LLM with new data, RAG offers a more economical way to augment the model's knowledge base.

The RAG process typically involves three main steps: Indexing, Retrieval, and Generation. Let's see how this application implements them.

**1. Indexing: Preparing the Knowledge Base**

This initial step involves processing the external knowledge (our guideline PDFs) and storing it in a way that's efficiently searchable.

* **Data Ingestion & Chunking:** When guideline PDFs are uploaded, the backend extracts the raw text using `pypdf` (`file_processing.extract_text_from_pdf`). This text is then broken down into smaller, manageable chunks using LangChain's text splitters (`file_processing.chunk_text`). This uses a two-pass approach (character splitting then token splitting) to create chunks suitable for embedding, respecting semantic boundaries where possible while adhering to token limits defined in `config.py`.
* **Embedding Generation:** Each text chunk needs to be converted into a numerical representation, called a vector embedding. This application uses a custom embedding function, `GeminiEmbeddingFunction` defined in `llm_service.py`, which utilizes Google's `models/text-embedding-004` model (configurable via `EMBEDDING_MODEL_NAME` in `config.py`). This function specifically sets the task type to `"retrieval_document"` for indexing.
    ```python
    # backend/app/services/llm_service.py
    from chromadb import EmbeddingFunction, Documents, Embeddings
    import google.generativeai as genai
    from app.core.config import EMBEDDING_MODEL_NAME, GOOGLE_API_KEY # Import config

    class GeminiEmbeddingFunction(EmbeddingFunction):
        def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, task_type: str = "retrieval_document"):
            self._model_name = model_name
            self._task_type = task_type
            # ... other setup ...

        # Simplified __call__ showing the core embedding logic
        def __call__(self, input_texts: Documents) -> Embeddings:
            if not GOOGLE_API_KEY: # Assuming GOOGLE_API_KEY check
                 # logger.error("Embedding call failed: GOOGLE_API_KEY not available.")
                 return [[] for _ in input_texts] # Handle missing key

            # Simplified call to underlying embedding generation (actual code uses retry)
            try:
                response = genai.embed_content(
                    model=self._model_name,
                    content=input_texts,
                    task_type=self._task_type # Task type used here
                )
                embeddings = response.get('embedding')
                if embeddings is None:
                     # logger.error(f"Embedding API call succeeded but returned no 'embedding' key.")
                     return [[] for _ in input_texts] # Handle API error
                return embeddings
            except Exception as e:
                 # logger.error(f"Error during embedding generation: {e}", exc_info=True)
                 return [[] for _ in input_texts] # Handle API error


        def set_task_type(self, task_type: str):
            valid_task_types = ["retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"]
            if task_type not in valid_task_types:
                 raise ValueError(f"Invalid task_type: {task_type}. Must be one of {valid_task_types}")
            self._task_type = task_type
            # logger.debug(f"GeminiEmbeddingFunction task_type set to: {task_type}")
    ```
* **Storing in ChromaDB:** The text chunks, along with their generated embeddings and metadata (like the source PDF filename), are stored in a **ChromaDB** vector database (`vector_store_service.py`). ChromaDB is an open-source vector store optimized for storing and querying embeddings based on semantic similarity.
    * The `VectorStoreService` initializes a persistent client (`chromadb.PersistentClient`) pointing to the path defined in `config.py`.
    * It uses `get_or_create_collection`, crucially passing our custom `GeminiEmbeddingFunction` instance to ensure consistency between indexing and querying embeddings.
    * The `add_documents` method adds data in batches for efficiency.
    ```python
    # backend/app/services/vector_store_service.py
    import chromadb
    from app.core.config import CHROMA_DB_PATH, COLLECTION_NAME
    from app.services.llm_service import GeminiEmbeddingFunction # Import custom function

    class VectorStoreService:
        # ... (singleton pattern, __init__) ...
        def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
            # ... initialize client ...
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.embed_fn = GeminiEmbeddingFunction() # Instantiate embedding function
            self._load_or_create_collection()
            # ...

        def _load_or_create_collection(self):
             self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # Pass the instance of our custom embedding function
                embedding_function=self.embed_fn
            )
            # ...

        def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
            if not self.collection: return False
            # Ensure the embedding function is set for indexing task
            self.embed_fn.set_task_type("retrieval_document")
            batch_size = 100 # Example batch size
            # ... (logging) ...
            try:
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    if not batch_chunks: continue
                    # Add batch to Chroma collection
                    self.collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                # ... (success logging) ...
                return True
            except Exception as e:
                # ... (error logging) ...
                return False
    ```
    The `DocumentProcessorService` orchestrates this by calling the relevant functions:
    ```python
    # backend/app/services/document_processor_service.py (in process_and_index_guidelines)
    # ... after extracting text and chunking ...
    # logger.info(f"Indexing {len(all_chunks_for_db)} total chunks...")
    # This call utilizes the VectorStoreService.add_documents method shown above
    if self.vector_store.add_documents(all_chunks_for_db, all_metadatas_for_db, all_ids_for_db):
        # ... success message ...
    else:
        # ... handle error ...
    ```

**2. Retrieval: Finding Relevant Information**

When the user asks a question (via chat) or submits an application for validation, the system needs to retrieve relevant information from the indexed guidelines.

* **Query Embedding:** The incoming user query or the text describing an application field is embedded using the *same* embedding model (`models/text-embedding-004`), but this time the `GeminiEmbeddingFunction`'s task type is set to `"retrieval_query"`. This ensures the query vector is comparable to the document vectors.
* **Similarity Search:** The `query_collection` method in `vector_store_service.py` takes the query text, embeds it (using the embedding function now set to "retrieval\_query"), and then uses ChromaDB's `collection.query()` method. This performs a similarity search (often using algorithms like Approximate Nearest Neighbors - ANN for speed) to find the `n_results` document embeddings in the database that are mathematically closest (e.g., using cosine similarity or Euclidean distance) to the query embedding.
* **Context Retrieval:** Crucially, the `query` method is configured with `include=['documents']` to retrieve the actual text chunks associated with the top-matching embeddings. This retrieved text forms the "context" that will be passed to the LLM.
    ```python
    # backend/app/services/vector_store_service.py
    def query_collection(self, query_text: str, n_results: int = 5) -> List[str]:
        if not self.collection or self.get_collection_count() == 0:
            # ... handle unavailable collection or empty collection ...
            return []

        # Set the embedding function task type specifically for querying
        self.embed_fn.set_task_type("retrieval_query")

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query_text], # The user's question/field description text
                n_results=n_results,      # Number of results to retrieve
                # Specify that we want the document text returned
                include=['documents']
            )
            # Extract the document texts from the results structure
            retrieved_docs = results['documents'][0] if results and results.get('documents') and results['documents'][0] else []
            return retrieved_docs
        except Exception as e:
            # logger.error(f"Error querying ChromaDB collection: {e}", exc_info=True)
            return []
    ```
    This is called within the `DocumentProcessorService` workflows:
    ```python
    # backend/app/services/document_processor_service.py (in validate_application_pdf)
    # For each field:
    query = f"What are the guidelines... for the field named: '{field_name}'...?"
    # This call utilizes the VectorStoreService.query_collection method shown above
    relevant_guidelines = self.vector_store.query_collection(query_text=query, n_results=5)
    # Now 'relevant_guidelines' holds the retrieved text chunks

    # backend/app/services/document_processor_service.py (in get_chat_response)
    context_query = f"Information related to: {user_query}"
    # This call also uses VectorStoreService.query_collection
    relevant_guidelines = self.vector_store.query_collection(query_text=context_query, n_results=5)
    # 'relevant_guidelines' holds context for the chat query
    ```

**3. Generation: Creating the Augmented Response**

This final step involves using the LLM, augmented with the retrieved context, to generate the final output.

* **Prompt Augmentation:** The retrieved guideline text chunks (`relevant_guidelines`) are incorporated into a carefully designed prompt along with the original user query or the application field details. The `llm_service.py` contains functions like `_build_validation_prompt` and the prompt logic within `generate_chat_response` that structure this input for the LLM.
    ```python
    # backend/app/services/llm_service.py (Example prompt structure in generate_chat_response)
    if not context_guidelines:
        # Prompt when no context is found
        prompt = f"""
        The user is asking: "{user_query}"
        I could not find any specific information for this query in the uploaded guideline documents.
        Please provide a helpful general response if appropriate, or clearly state that the information is not available in the documents.
        Be concise and polite. Do not invent information about guidelines if none were found.
        """
    else:
        # Prompt when context IS found
        guideline_context_str = "\\n\\n---\\n\\n".join(context_guidelines) # Joining retrieved chunks
        prompt = f"""
        You are a helpful assistant answering questions based ONLY on the provided "Guideline Context".
        If the context does not contain information relevant to the user's question, clearly state that you cannot find the answer in the provided documents.
        Do NOT make up information or use external knowledge. Be concise.

        Guideline Context:
        --- START CONTEXT ---
        {guideline_context_str} # Injecting the retrieved context
        --- END CONTEXT ---

        User Query: "{user_query}"

        Your Answer (based *only* on the context above):
        """
    ```
* **LLM Generation:** This augmented prompt is then sent to the generative LLM (e.g., Gemini `gemini-1.5-flash` configured via `GENERATIVE_MODEL_NAME` in `config.py`) using the `generate_content` method. The LLM uses the provided context as its primary source of information to generate the validation decision, the chat response, or the formatted summary requested by the specific prompt.
    ```python
    # backend/app/services/llm_service.py (in generate_chat_response)
    try:
        # logger.info(f"Generating chat response for query: '{user_query[:100]}...'")
        # Calling the generative model with the augmented prompt
        response = generative_model.generate_content( # generative_model is the initialized Gemini model
            prompt, # The prompt containing the query and retrieved context
            generation_config=genai_types.GenerationConfig(temperature=0.2) # Control creativity
        )
        return response.text.strip() # Return the LLM's generated text
    except Exception as e:
        # logger.error(f"Error during LLM chat response generation: {e}", exc_info=True)
        return f"An error occurred while generating the chat response: {str(e)}"
    ```

By following this Index -> Retrieve -> Generate process, the RAG architecture allows the application to provide accurate, context-aware responses grounded in the specific guideline documents provided by the user, significantly enhancing the LLM's capabilities for this task.

## Technologies Used

The detailed review confirms the stack:

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **LLM:** Google Gemini (e.g., `gemini-1.5-flash` for generation, `models/text-embedding-004` for embeddings)
* **Orchestration/Utils:** LangChain (specifically `langchain-text-splitters`), Custom Service Logic
* **Vector Store:** ChromaDB
* **PDF Parsing:** pypdf
* **Language:** Python

## Conclusion

This Application Validation Assistant showcases a well-structured approach to building RAG applications. By separating concerns into distinct services for file processing, vector storage, LLM interaction, and overall orchestration, managed by a FastAPI backend and presented through a Streamlit frontend, it creates a powerful and maintainable tool. The specific use of custom embedding functions tailored for Gemini, detailed prompting strategies, and multi-step processing (extraction -> retrieval -> validation/generation -> formatting) highlights key techniques for effective RAG implementation. It provides a practical solution to automate the comparison of documents against guidelines, saving time and improving consistency.