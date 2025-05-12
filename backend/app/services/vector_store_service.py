import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings # Already imported in llm_service but good here too
from typing import List, Optional, Dict, Any
from tqdm.auto import tqdm

# from backend.app.core.config import CHROMA_DB_PATH, COLLECTION_NAME, get_logger
# from backend.app.services.llm_service import GeminiEmbeddingFunction # Import the custom embedding function

from app.core.config import CHROMA_DB_PATH, COLLECTION_NAME, get_logger
from app.services.llm_service import GeminiEmbeddingFunction # Import the custom embedding function

logger = get_logger(__name__)

class VectorStoreService:
    """
    Manages interactions with the ChromaDB vector store.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
        if hasattr(self, '_initialized') and self._initialized: # Prevent re-initialization for singleton
            return
        
        logger.info(f"Initializing VectorStoreService with DB path: {db_path} and collection: {collection_name}")
        self.db_path = db_path
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embed_fn = GeminiEmbeddingFunction() # Instantiate our custom embedding function

        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self._load_or_create_collection()
            logger.info("VectorStoreService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            self.client = None # Ensure client is None on failure
            self.collection = None
        
        self._initialized = True


    def _load_or_create_collection(self):
        """Loads an existing collection or creates it if it doesn't exist."""
        if not self.client:
            logger.error("ChromaDB client not initialized. Cannot load/create collection.")
            raise RuntimeError("ChromaDB client is not available.")
        try:
            # Check if collection exists
            # collections = self.client.list_collections()
            # if any(c.name == self.collection_name for c in collections):
            #     logger.info(f"Loading existing collection: {self.collection_name}")
            #     self.collection = self.client.get_collection(name=self.collection_name, embedding_function=self.embed_fn)
            # else:
            #     logger.info(f"Creating new collection: {self.collection_name}")
            #     self.collection = self.client.create_collection(
            #         name=self.collection_name,
            #         embedding_function=self.embed_fn
            #     )
            # For simplicity with clearing, let's always try to get_or_create
            # The clearing logic will be explicit via clear_collection()
            logger.info(f"Getting or creating collection: {self.collection_name}")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn # type: ignore [arg-type] # ChromaDB type hint issue
            )
            logger.info(f"Collection '{self.collection_name}' ready. Current count: {self.get_collection_count()}")

        except Exception as e:
            logger.error(f"Error getting or creating collection '{self.collection_name}': {e}", exc_info=True)
            self.collection = None # Ensure collection is None on failure
            raise # Re-raise the exception to signal failure

    def clear_collection(self) -> bool:
        """Clears all documents from the collection by deleting and recreating it."""
        if not self.client:
            logger.error("ChromaDB client not initialized. Cannot clear collection.")
            return False
        try:
            logger.info(f"Attempting to delete collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted. Recreating...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn # type: ignore [arg-type]
            )
            logger.info(f"Collection '{self.collection_name}' cleared and recreated successfully.")
            return True
        except Exception as e:
            # If deletion fails because collection doesn't exist, that's okay for a "clear" operation.
            # We can then try to create it.
            logger.warning(f"Could not delete collection '{self.collection_name}' (it might not exist): {e}. Attempting to create.")
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embed_fn # type: ignore [arg-type]
                )
                logger.info(f"Collection '{self.collection_name}' created after failed delete (assumed clear).")
                return True
            except Exception as create_e:
                logger.error(f"Failed to recreate collection '{self.collection_name}' after delete attempt: {create_e}", exc_info=True)
                self.collection = None
                return False


    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """
        Adds documents (chunks) with their metadatas and IDs to the collection.
        Uses the 'retrieval_document' task type for embeddings.
        """
        if not self.collection:
            logger.error("Collection not available. Cannot add documents.")
            return False
        if not chunks:
            logger.info("No chunks provided to add_documents.")
            return True # No error, just nothing to do

        self.embed_fn.set_task_type("retrieval_document") # Ensure correct task type for indexing
        
        batch_size = 100  # ChromaDB recommendation for batching
        logger.info(f"Adding {len(chunks)} documents to collection '{self.collection_name}' in batches of {batch_size}.")

        try:
            for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing Chunks to ChromaDB"):
                batch_chunks = chunks[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]

                if not batch_chunks:
                    continue
                
                # logger.debug(f"Adding batch {i//batch_size + 1} ({len(batch_chunks)} chunks)...")
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            logger.info(f"Successfully added {len(chunks)} documents to collection.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            return False

    def query_collection(self, query_text: str, n_results: int = 5) -> List[str]:
        """
        Queries the collection to find relevant document chunks.
        Uses the 'retrieval_query' task type for embeddings.
        """
        if not self.collection:
            logger.error("Collection not available. Cannot query.")
            return []
        
        if self.get_collection_count() == 0:
            logger.warning("Querying an empty collection. No results will be found.")
            return []

        self.embed_fn.set_task_type("retrieval_query") # Ensure correct task type for querying
        
        try:
            # logger.debug(f"Querying collection for: '{query_text[:100]}...' (n_results={n_results})")
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents']  # We only need the document content for context
            )
            
            retrieved_docs = results['documents'][0] if results and results.get('documents') and results['documents'][0] else []
            # logger.debug(f"Retrieved {len(retrieved_docs)} guideline chunks for query.")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error querying ChromaDB collection: {e}", exc_info=True)
            return []

    def get_collection_count(self) -> int:
        """Returns the number of items in the collection."""
        if self.collection:
            try:
                return self.collection.count()
            except Exception as e:
                logger.error(f"Error getting collection count: {e}", exc_info=True)
                return 0
        return 0

# Global instance for easy access, or use dependency injection in FastAPI
vector_store_service_instance = VectorStoreService()
