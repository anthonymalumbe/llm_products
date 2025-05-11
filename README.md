# RAG Application Assistant

This project implements a Retrieval Augmented Generation (RAG) system to assist with understanding and validating application forms against a set of guideline documents. It consists of a FastAPI backend for document processing and AI interaction, and a Streamlit frontend for user interaction.

## Features

* **Guideline Processing**: Upload PDF guideline documents. The backend extracts text, chunks it, generates embeddings, and stores them in a ChromaDB vector store.
* **Application Validation**: Upload an application form (PDF). The backend extracts structured data from the form using an LLM, then validates each field against the indexed guidelines using a RAG approach with an LLM. Results are formatted for readability.
* **Chat with Guidelines**: Interact with a chatbot that answers questions based on the content of the processed guideline documents.
* **Modular Backend**: FastAPI backend structured with services, routers, schemas, and utilities for better maintainability.
* **Clear Frontend-Backend Separation**: Streamlit UI communicates with the FastAPI backend via HTTP requests.

## Project Structure
```text
rag_application_assistant/
├── backend/
│   ├── app/
│   │   ├── init.py
│   │   ├── main.py             # FastAPI app instantiation, include routers
│   │   ├── api/
│   │   │   ├── init.py
│   │   │   └── endpoints.py    # FastAPI routers and endpoint logic
│   │   ├── services/
│   │   │   ├── init.py
│   │   │   ├── llm_service.py    # Gemini model interactions
│   │   │   ├── vector_store_service.py # ChromaDB interactions
│   │   │   └── document_processor_service.py # Orchestration logic
│   │   ├── schemas/
│   │   │   ├── init.py
│   │   │   └── models.py       # Pydantic models
│   │   ├── core/
│   │   │   ├── init.py
│   │   │   └── config.py       # Configuration variables
│   │   └── utils/
│   │       ├── init.py
│   │       └── file_processing.py # PDF extraction, text chunking
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── app.py                  # Streamlit application
│   ├── requirements.txt        # Frontend dependencies
|   └── images        # Frontend dependencies
├── data/                       # (To be created by user for local testing if desired)
│   ├── guidance/
│   └── pre_submitted_form/
├── vector_store/               # (Created by backend for ChromaDB persistence)
│   └── chroma_db/
├── .gitignore                  # (Recommended: add venv, pycache, .DS_Store, etc.)
└── README.md
```

## Prerequisites

* Python 3.8+
* Google Cloud Project with the Generative Language API (Gemini) enabled.
* A `GOOGLE_API_KEY` environment variable set with your API key.

## Setup and Installation
First, let’s install uv and set up our Python project and environment:



1.  **Clone the repository (if applicable):**
    ```bash
    git clone [this Repo](https://github.com/anthonymalumbe/llm_products)
    cd rag_application_assistant
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```MacOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    ```MacOS/Linux
    # Create a new directory for our project
    uv init rag_application_assistant
    cd rag_application_assistant
    
    # Create virtual environment and activate it
    uv venv
    source .venv/bin/activate    
    ```

3.  **Create necessary data directories (optional, for local file persistence if backend saves them):**
    ```bash
    mkdir -p ./data/guidance ./data/pre_submitted_form
    ```
    The `./vector_store/chroma_db/` directory will be created automatically by the backend.

4.  **Set your Google API Key:**
    Ensure the `GOOGLE_API_KEY` environment variable is set. Refer to your OS documentation.
    Example (Linux/macOS): `export GOOGLE_API_KEY='YOUR_API_KEY'`

5.  **Install dependencies:**
    * **For the backend:**
        ```bash
        cd backend
        pip install -r requirements.txt
        or
        uv add fastapi uvicorn[standard] pydantic python-dotenv google-generativeai google-api-core chromadb pypdf pypdf2 tqdm langchain-text-splitters sentence-transformers python-multipart cryptography 
        cd ..
        ```
    * **For the frontend:**
        ```bash
        cd frontend
        pip install -r requirements.txt
        or uv add streamlit requests pandas
        cd ..
        ```

## Running the Application

You need to run the backend (FastAPI) and the frontend (Streamlit) separately, typically in two different terminal sessions (with the virtual environment activated in both).

### 1. Run the Backend (FastAPI)

Navigate to the `backend` directory:
```bash
cd backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

The backend API will be available at http://localhost:8000. The endpoints are prefixed with /api/v1/.
```

### 2. Run the Frontend (Streamlit)

Navigate to the frontend directory:
```bash
cd frontend
streamlit run app.py
```

The Streamlit application will typically open in your web browser automatically (e.g., http://localhost:8501). It is configured to communicate with the backend at http://localhost:8000.

If your backend runs on a different URL/port, set the BACKEND_URL environment variable before running Streamlit:

```bash
# Example for Linux/macOS
export BACKEND_URL='http://your-custom-backend-url:port'
streamlit run app.py
```

Usage
Open the Streamlit application in your browser.
The frontend will attempt to connect to the backend. Status messages might appear.
Upload Guideline PDFs: In the "Process Guidelines" section, upload PDF files. Click "Process Selected Guidelines".
Upload Application PDF: Once guidelines are processed, upload an application PDF in the "Validate Application" section. Click "Validate Application Form".
View Validation Results: The validation summary will be displayed.
Chat with Guidelines: Use the chat interface to ask questions about the processed guidelines.
Key Configuration
Backend: Most configurations (API keys, model names, paths) are in backend/app/core/config.py. Many can be overridden by environment variables.
Frontend: The BACKEND_URL for the API is configured in frontend/app.py and can be overridden by an environment variable.
Development Notes
The backend services (llm_service.py, vector_store_service.py, document_processor_service.py) encapsulate specific functionalities.
FastAPI's dependency injection can be further utilized for managing service instances if needed.
Error handling has been improved, but can always be made more granular.

---
### `.gitignore` (Recommended)
