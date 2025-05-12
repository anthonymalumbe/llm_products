# backend/app/services/llm_service.py
import time
import ast
import json
import re
import textwrap
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings

from app.core.config import (
    GOOGLE_API_KEY, EMBEDDING_MODEL_NAME, GENERATIVE_MODEL_NAME,
    RETRY_CODES, get_logger
)

# from backend.app.core.config import (
#     GOOGLE_API_KEY, EMBEDDING_MODEL_NAME, GENERATIVE_MODEL_NAME,
#     RETRY_CODES, get_logger
# )

logger = get_logger(__name__)

# --- Initialize Generative Model ---
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        generative_model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        logger.info(f"Generative Model '{GENERATIVE_MODEL_NAME}' initialized in llm_service.")
    except Exception as e:
        logger.error(f"Failed to initialize Generative Model in llm_service: {e}", exc_info=True)
        generative_model = None
else:
    logger.warning("GOOGLE_API_KEY not found. Generative Model in llm_service is not initialized.")
    generative_model = None


# --- Embedding Generation with Retry ---
@retry.Retry(predicate=lambda e: isinstance(e, Exception) and (
    (isinstance(e, genai_types.generation_types.StopCandidateException)) or
    # (isinstance(e, genai.errors.ResourceExhaustedError)) or # genai.errors doesn't exist directly
    # (isinstance(e, genai.errors.GoogleAPICallError) and e.code in RETRY_CODES)
    # Simplified retry predicate, check Google API core exceptions if more specific needed
    (hasattr(e, 'code') and e.code in RETRY_CODES) if hasattr(e, 'code') else False
))
def generate_embeddings_with_retry(model_name: str, task_type: str, texts: List[str]) -> Optional[List[List[float]]]:
    """
    Generates embeddings with retry logic.

    Args:
        model_name (str): The name of the embedding model.
        task_type (str): The task type for embedding.
        texts (List[str]): List of texts to embed.

    Returns:
        Optional[List[List[float]]]: List of embedding vectors, or None on failure.
    """
    if not GOOGLE_API_KEY:
        logger.error("Cannot generate embeddings: GOOGLE_API_KEY not configured.")
        return None
    if not texts:
        logger.warning("generate_embeddings_with_retry called with empty text list.")
        return []
    try:
        logger.debug(f"Generating {len(texts)} embeddings with model {model_name} for task {task_type}.")
        response = genai.embed_content(
            model=model_name,
            content=texts,
            task_type=task_type
        )
        embeddings = response.get('embedding')
        if embeddings is None:
            logger.error(f"Embedding API call succeeded but returned no 'embedding' key. Response: {response}")
            return None
        # logger.debug(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Error during embedding generation (model: {model_name}, task: {task_type}): {e}", exc_info=True)
        time.sleep(2) # Delay before potential retry
        # The @retry decorator will handle raising the exception if retries are exhausted.
        # However, if we want to return None instead of raising, we should catch it here.
        # For now, let retry handle re-raising. If it makes it past retry, it was successful.
        # If all retries fail, the exception will propagate. We might want to catch it upstream.
        raise # Re-raise to be caught by retry or calling function

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function using Google Gemini."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, task_type: str = "retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
        if not GOOGLE_API_KEY:
            logger.error("GeminiEmbeddingFunction initialized without GOOGLE_API_KEY. Embeddings will fail.")

    def __call__(self, input_texts: Documents) -> Embeddings:
        """
        Generates embeddings for the given input documents.
        """
        if not GOOGLE_API_KEY:
            logger.error("Embedding call failed: GOOGLE_API_KEY not available.")
            # ChromaDB expects a list of embeddings, even if empty or erroneous.
            # Returning empty lists or lists of zeros might be one way to handle this.
            # For simplicity, let's raise an error if it's critical.
            # However, ChromaDB might not handle exceptions well in __call__.
            # It's safer to return empty embeddings of the correct dimension if possible, or fail gracefully.
            return [[] for _ in input_texts] # Return list of empty lists

        embeddings = generate_embeddings_with_retry(
            model_name=self._model_name,
            task_type=self._task_type,
            texts=input_texts
        )
        if embeddings is None: # Handle failure from generate_embeddings_with_retry
             logger.error(f"Failed to generate embeddings for {len(input_texts)} documents. Returning empty embeddings.")
             return [[] for _ in input_texts] # Or handle error more explicitly
        return embeddings

    def set_task_type(self, task_type: str):
        """Allows switching task type, e.g., for queries."""
        valid_task_types = ["retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"]
        if task_type not in valid_task_types:
             raise ValueError(f"Invalid task_type: {task_type}. Must be one of {valid_task_types}")
        self._task_type = task_type
        logger.debug(f"GeminiEmbeddingFunction task_type set to: {task_type}")

# --- LLM-based Data Extraction ---
def create_prompt_for_dict_extraction(form_text: str) -> str:
    """Creates a prompt for LLM to extract key-value pairs as a Python dictionary."""
    return f"""
    Extract all input fields and their corresponding values from the following form text.
    Return ONLY a valid Python dictionary where keys are field names and values are the extracted inputs.
    Do not include any explanations, markdown formatting (like ```python), or any text outside the dictionary itself.

    Example format:
    {{
        "Applicant Name": "John Doe",
        "Date of Birth": "1990-01-15",
        "Has Driving License": true,
        "Address": "123 Main St, Anytown, USA"
    }}

    Form text:
    ---
    {form_text}
    ---
    Extracted Dictionary:
    """

def generate_dictionary_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Uses LLM to extract a dictionary from text."""
    if not generative_model:
        logger.error("LLM for dictionary extraction not available.")
        return None
    prompt = create_prompt_for_dict_extraction(text)
    try:
        response = generative_model.generate_content(
            prompt,
            generation_config=genai_types.GenerationConfig(temperature=0.0) # Low temp for structured output
        )
        output_text = response.text.strip()
        
        # Try to find the dictionary within potential markdown
        match = re.search(r"(\{[\s\S]*?\})", output_text)
        if match:
            dict_str = match.group(1)
        else:
            dict_str = output_text # Assume raw output is the dict if no clear block found

        try:
            extracted_data = ast.literal_eval(dict_str)
            if not isinstance(extracted_data, dict):
                logger.error(f"LLM output for dict extraction was not a dictionary: {type(extracted_data)}. Output: {output_text}")
                return None
            logger.info(f"Successfully extracted dictionary with {len(extracted_data)} items from text.")
            return extracted_data
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error parsing dictionary from LLM response: {e}. Raw output: '{output_text}'", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"Error during LLM call for dictionary extraction: {e}", exc_info=True)
        return None

# --- LLM-based Validation ---
def _build_validation_prompt(field_name: str, field_value: Any, guidelines: List[str]) -> str:
    """Constructs the prompt for validating a single application field."""
    guideline_context = "\n\n---\n\n".join(guidelines) if guidelines else "No specific guidelines provided for this field."
    return f"""
        Analyze the application form entry below based on the provided "Guideline Context".

        Application Field: "{field_name}"
        Value Submitted: "{field_value}"

        Guideline Context:
        --- START CONTEXT ---
        {guideline_context}
        --- END CONTEXT ---

        Instructions:
        1. Carefully interpret the "Application Field" and its "Value Submitted".
        2. Determine if the "Value Submitted" is VALID or INVALID according to the "Guideline Context".
        3. If the context is insufficient or doesn't cover this specific field, respond with CANNOT DETERMINE.
        4. Respond *only* with the following format (no extra text, no markdown):

        VALIDATION_STATUS: [VALID/INVALID/CANNOT DETERMINE]
        EXPLANATION: [A brief explanation for your decision, strictly based on the Guideline Context. Quote relevant parts of the context if possible. If CANNOT DETERMINE, explain why the context is insufficient.]

        Example VALID:
        VALIDATION_STATUS: VALID
        EXPLANATION: The value 'Yes' is acceptable as the context states 'Section A requires an affirmative response'.

        Example INVALID:
        VALIDATION_STATUS: INVALID
        EXPLANATION: The value '17' is not acceptable because the guideline context specifies 'Applicants must be 18 years or older'.

        Example CANNOT DETERMINE:
        VALIDATION_STATUS: CANNOT DETERMINE
        EXPLANATION: The provided Guideline Context does not contain specific rules or mention requirements for the '{field_name}' field.
    """

def validate_entry_with_llm(field_name: str, field_value: Any, guidelines: List[str]) -> Dict[str, Any]:
    """Uses LLM with context to validate an application entry."""
    if not generative_model:
        logger.error(f"LLM for validation not available. Cannot validate '{field_name}'.")
        return {"field": field_name, "value": field_value, "isValid": None, "reason": "Validation service (LLM) unavailable."}

    logger.info(f"Validating (LLM): '{field_name}' = '{str(field_value)[:50]}...'")

    prompt = _build_validation_prompt(field_name, field_value, guidelines)
    default_uncertain_reason = f"No relevant guidelines were found or provided to validate the field '{field_name}'."
    
    if not guidelines: # If retriever found nothing, don't even call LLM, just mark as uncertain
        logger.warning(f"No guidelines provided for '{field_name}'. Marking as CANNOT DETERMINE without LLM call.")
        return {
            "field": field_name, "value": field_value, "isValid": None,
            "reason": default_uncertain_reason
        }

    try:
        response = generative_model.generate_content(
            prompt,
            generation_config=genai_types.GenerationConfig(temperature=0.1)
        )
        raw_text = response.text.strip()
        # logger.debug(f"LLM validation raw response for '{field_name}':\n{raw_text}")

        status_match = re.search(r"VALIDATION_STATUS\s*:\s*(VALID|INVALID|CANNOT\s*DETERMINE)", raw_text, re.IGNORECASE)
        explanation_match = re.search(r"EXPLANATION\s*:\s*([\s\S]+)", raw_text, re.IGNORECASE)

        if not status_match:
            logger.warning(f"Could not parse VALIDATION_STATUS from LLM response for '{field_name}'. Raw: {raw_text}")
            return {"field": field_name, "value": field_value, "isValid": None, "reason": f"LLM response format error: {raw_text}"}

        status_str = status_match.group(1).upper().replace(" ", "_")
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided by LLM."
        
        is_valid_map = {"VALID": True, "INVALID": False, "CANNOT_DETERMINE": None}
        is_valid = is_valid_map.get(status_str)

        return {"field": field_name, "value": field_value, "isValid": is_valid, "reason": explanation}

    except Exception as e:
        logger.error(f"Exception during LLM validation for '{field_name}': {e}", exc_info=True)
        return {"field": field_name, "value": field_value, "isValid": None, "reason": f"LLM validation error: {str(e)}"}

# --- LLM-based Formatting ---
def format_validation_results_with_llm(validation_results: List[Dict[str, Any]]) -> str:
    """Uses LLM to format validation results into a human-readable summary."""
    if not generative_model:
        logger.error("LLM for formatting not available.")
        return "Error: Formatting service (LLM) unavailable."
    if not validation_results:
        return "No validation results to format."

    results_json_str = json.dumps(validation_results, indent=2)
    prompt = f"""
        Given the following validation results in JSON format, produce a concise, human-readable summary.
        Use the specified status icons (✅, ❌, ❓).
        **IMPORTANT:** Do NOT use markdown headers (like #, ##). Use simple text and '---' as a separator between entries.
        For the 'Reason' of each entry, try to put each sentence on a new line.
        Conclude with a 'Summary' section listing total counts for Passed, Failed, and Uncertain.

        JSON Results:
        ```json
        {results_json_str}
        ```

        Desired Output Format (example):
        ✅ Field : [fieldName]
        Value : [valueSubmitted]
        Status: Passed
        Reason: [reason, sentences on new lines]
        ---
        ❌ Field : [fieldName]
        Value : [valueSubmitted]
        Status: Failed
        Reason: [reason, sentences on new lines]
        ---
        ❓ Field : [fieldName]
        Value : [valueSubmitted]
        Status: Uncertain
        Reason: [reason, sentences on new lines]
        ---
        ... (repeat for all entries) ...

        Summary
        ✅ Passed   : [Count]
        ❌ Failed   : [Count]
        ❓ Uncertain: [Count]
    """
    try:
        logger.info(f"Requesting LLM to format {len(validation_results)} validation results.")
        response = generative_model.generate_content(
            prompt,
            generation_config=genai_types.GenerationConfig(temperature=0.1)
        )
        formatted_text = response.text.strip()
        logger.info("Successfully formatted validation results using LLM.")
        return formatted_text
    except Exception as e:
        logger.error(f"Error during LLM formatting of results: {e}", exc_info=True)
        # Fallback basic formatting
        fallback_text = "Validation Results Summary (LLM Formatting Failed):\n\n"
        passed = sum(1 for r in validation_results if r.get('isValid') is True)
        failed = sum(1 for r in validation_results if r.get('isValid') is False)
        uncertain = sum(1 for r in validation_results if r.get('isValid') is None)

        for res in validation_results:
            icon = "❓"
            status = "Uncertain"
            if res.get('isValid') is True: icon, status = "✅", "Passed"
            elif res.get('isValid') is False: icon, status = "❌", "Failed"
            
            reason_lines = textwrap.fill(res.get('reason', 'N/A'), width=70, initial_indent="  ", subsequent_indent="  ")
            fallback_text += f"{icon} Field : {res.get('field', 'N/A')}\n"
            fallback_text += f"Value : {str(res.get('value', 'N/A'))[:100]}\n" # Truncate long values
            fallback_text += f"Status: {status}\n"
            fallback_text += f"Reason:\n{reason_lines}\n---\n"
        
        fallback_text += f"\nSummary\n✅ Passed   : {passed}\n❌ Failed   : {failed}\n❓ Uncertain: {uncertain}\n"
        return fallback_text

# --- LLM-based Chat Response ---
def generate_chat_response(user_query: str, context_guidelines: List[str]) -> str:
    """Generates a chat response using LLM with provided context."""
    if not generative_model:
        logger.error("LLM for chat not available.")
        return "Error: Chat service (LLM) unavailable."

    if not context_guidelines:
        logger.info(f"No relevant guidelines found for query: '{user_query}'. Responding generally.")
        prompt = f"""
        The user is asking: "{user_query}"
        I could not find any specific information for this query in the uploaded guideline documents.
        Please provide a helpful general response if appropriate, or clearly state that the information is not available in the documents.
        Be concise and polite. Do not invent information about guidelines if none were found.
        """
    else:
        guideline_context_str = "\n\n---\n\n".join(context_guidelines)
        prompt = f"""
        You are a helpful assistant answering questions based ONLY on the provided "Guideline Context".
        If the context does not contain information relevant to the user's question, clearly state that you cannot find the answer in the provided documents.
        Do NOT make up information or use external knowledge. Be concise.

        Guideline Context:
        --- START CONTEXT ---
        {guideline_context_str}
        --- END CONTEXT ---

        User Query: "{user_query}"

        Your Answer (based *only* on the context above):
        """
    try:
        logger.info(f"Generating chat response for query: '{user_query[:100]}...'")
        response = generative_model.generate_content(
            prompt,
            generation_config=genai_types.GenerationConfig(temperature=0.2)
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error during LLM chat response generation for query '{user_query}': {e}", exc_info=True)
        return f"An error occurred while generating the chat response: {str(e)}"