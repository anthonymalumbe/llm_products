# backend/app/utils/file_processing.py
import re
import logging
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from backend.app.core.config import (
    CHARACTER_CHUNK_SIZE, CHARACTER_CHUNK_OVERLAP,
    TOKEN_CHUNK_SIZE, TOKEN_CHUNK_OVERLAP, get_logger
)

logger = get_logger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a PDF file.
    Each page's text is prefixed with "--- Page [page_number+1] ---".

    Args:
        pdf_path (str): The file system path to the PDF file.

    Returns:
        str: A single string containing all extracted text from the PDF.
             Returns an empty string if an error occurs or no text is found.
    """
    logger.info(f"Extracting text from: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        pdf_text = ""
        if not reader.pages:
            logger.warning(f"No pages found in PDF: {pdf_path}")
            return ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                cleaned_text = text.strip()
                pdf_text += f"\n--- Page {i+1} ---\n{cleaned_text}"
            else:
                logger.warning(f"No text extracted from page {i+1} of {pdf_path}")
        if not pdf_text:
             logger.warning(f"No text extracted from any page in PDF: {pdf_path}")
        else:
            logger.info(f"Successfully extracted text from {pdf_path} ({len(reader.pages)} pages).")
        return pdf_text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}", exc_info=True)
        return ""

def word_wrap(string: str, n_chars: int = 72) -> str:
    """
    Wraps a string at the next space after a specified number of characters.
    This is a simple recursive word wrap.

    Args:
        string (str): The string to wrap.
        n_chars (int, optional): The maximum number of characters per line before wrapping.
                                 Defaults to 72.

    Returns:
        str: The word-wrapped string.
    """
    if len(string) <= n_chars:
        return string
    else:
        wrap_at = string.rfind(' ', 0, n_chars)
        if wrap_at == -1: # No space found, force break
            wrap_at = n_chars
        return string[:wrap_at] + '\n' + word_wrap(string[wrap_at:].lstrip(), n_chars)

def chunk_text(
    text: str,
    character_chunk_size: int = CHARACTER_CHUNK_SIZE,
    character_chunk_overlap: int = CHARACTER_CHUNK_OVERLAP,
    token_chunk_size: int = TOKEN_CHUNK_SIZE,
    token_chunk_overlap: int = TOKEN_CHUNK_OVERLAP
) -> List[str]:
    """
    Splits a given text into overlapping chunks suitable for embedding.
    It performs a two-pass splitting:
    1. RecursiveCharacterTextSplitter: Splits based on characters, respecting sentence/paragraph boundaries.
    2. SentenceTransformersTokenTextSplitter: Further splits character-wise chunks based on token counts.

    Args:
        text (str): The input text to be chunked.
        character_chunk_size (int, optional): The target size for character-based chunks.
        character_chunk_overlap (int, optional): The overlap between character-based chunks.
        token_chunk_size (int, optional): The target size (in tokens) for token-based chunks.
        token_chunk_overlap (int, optional): The overlap (in tokens) between token-based chunks.

    Returns:
        List[str]: A list of text chunks. Returns an empty list if the input text is empty.
    """
    if not text:
        logger.warning("Attempted to chunk empty text.")
        return []
    # First pass: character‐aware splitting
    char_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=character_chunk_size,
        chunk_overlap=character_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    char_chunks = char_splitter.split_text(text)
    if not char_chunks:
        logger.warning(f"Character splitting produced no chunks for text starting with: {text[:100]}...")
        # return [] # If character splitting yields nothing, token splitting won't help
    
    # Second pass: token‐aware splitting
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=token_chunk_overlap,
        tokens_per_chunk=token_chunk_size,
    )
    
    all_token_chunks: List[str] = []
    for char_chunk in char_chunks:
        if not char_chunk.strip(): # Skip empty or whitespace-only chunks
            continue
        try:
            token_chunks_from_char_chunk = token_splitter.split_text(char_chunk)
            all_token_chunks.extend(token_chunks_from_char_chunk)
        except Exception as e: # Catch potential errors from token_splitter if a chunk is problematic
            logger.error(f"Error during token splitting for chunk: '{char_chunk[:100]}...': {e}", exc_info=True)
            # Optionally, add the char_chunk itself if token splitting fails, or skip
            # all_token_chunks.append(char_chunk) # Add original char_chunk as fallback

    if not all_token_chunks and char_chunks:
        # If token splitting yielded nothing but char_chunks existed, consider using char_chunks
        logger.warning("Token splitting produced no chunks, but character chunks existed. This might indicate very short character chunks.")
        # return [re.sub(r'\s+', ' ', c).strip() for c in char_chunks if c.strip()] # Fallback to char_chunks

    cleaned_chunks = [re.sub(r'\s+', ' ', c).strip() for c in all_token_chunks]
    final_chunks = [c for c in cleaned_chunks if c] # Filter out any empty strings after cleaning

    if not final_chunks:
        logger.warning(f"No valid chunks generated after cleaning for text starting with: {text[:100]}...")
    else:
        logger.info(f"Generated {len(final_chunks)} chunks from text.")
    return final_chunks