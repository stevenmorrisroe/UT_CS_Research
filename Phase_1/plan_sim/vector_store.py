import logging
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Constants
QDRANT_PATH = "qdrant_db"
QDRANT_COLLECTION = "ideas_collection"
OPENAI_EMBEDDING_DIM = 1536 # Assuming this is standard for OpenAI embeddings
IDEA_SIMILARITY_THRESHOLD = 0.8

# --- Qdrant Client Initialization ---
client = QdrantClient(path=QDRANT_PATH)
try:
    client.get_collection(QDRANT_COLLECTION)
    logging.info(f"Connected to existing Qdrant collection: {QDRANT_COLLECTION}")
except Exception: # Catch broader exception for collection not existing or other issues
    logging.info(f"Creating Qdrant collection: {QDRANT_COLLECTION}")
    try:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=OPENAI_EMBEDDING_DIM,
                distance=models.Distance.COSINE
            )
        )
        logging.info(f"Successfully created Qdrant collection: {QDRANT_COLLECTION}")
    except Exception as create_error:
        logging.error(f"Failed to create Qdrant collection {QDRANT_COLLECTION}: {create_error}", exc_info=True)
        # Depending on the application, might want to raise here or handle differently
        raise

# Instantiate QdrantVectorStore
# Ensure embeddings are configured (using OpenAIEmbeddings here)
embeddings = OpenAIEmbeddings() 
qdrant_vector_store = QdrantVectorStore(
    client=client, 
    collection_name=QDRANT_COLLECTION, 
    embedding=embeddings
)
logging.info("QdrantVectorStore initialized.")
# --- End Qdrant Initialization ---

def is_idea_novel(idea_text: str, step_number: int, threshold: float = IDEA_SIMILARITY_THRESHOLD) -> bool:
    """Check if an idea is novel by searching Qdrant for similar ideas at the same step."""
    metadata_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="step",
                match=models.MatchValue(value=step_number)
            )
        ]
    )
    try:
        # Use the QdrantVectorStore instance for similarity search
        similar_entries = qdrant_vector_store.similarity_search_with_score( # Use with_score for clarity
            query=idea_text,
            k=1, # Find the closest match
            filter=metadata_filter
        )
        
        # Check if any similar entry meets the threshold
        if similar_entries and similar_entries[0][1] >= threshold: # Score is second element
             logging.debug(f"Found similar idea for step {step_number} with score {similar_entries[0][1]} >= {threshold}.")
             return False # Not novel
        
        logging.debug(f"No sufficiently similar idea found for step {step_number}. Novelty confirmed.")
        return True # Novel
        
    except Exception as e:
        logging.error(f"Error during Qdrant similarity search for step {step_number}: {e}", exc_info=True)
        # Fail safe: assume not novel on error to avoid potential duplicates
        logging.warning(f"Assuming idea is not novel for step {step_number} due to search error.")
        return False

def store_idea(idea_text: str, step_number: int):
    """Stores the generated idea text in Qdrant with its step number."""
    doc = Document(page_content=idea_text, metadata={"step": step_number})
    try:
        # Use add_documents method of QdrantVectorStore
        qdrant_vector_store.add_documents([doc])
        logging.info(f"Stored idea in Qdrant for step {step_number}.")
    except Exception as e:
        logging.error(f"Error storing idea in Qdrant for step {step_number}: {e}", exc_info=True)
        # Depending on requirements, might want to raise this error 