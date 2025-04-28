"""
Utilities for product embedding and semantic search within product indices.

Provides functions to:
- Lazily load embedding models (Google Generative AI) and libraries (Pandas, Scikit-learn).
- Embed single product descriptions or lists of descriptions.
- Search a CSV product index (containing descriptions and ranks) using a query embedding.
- Returns the average rank of the top K most similar products found in the index.
"""
# core/product_search.py
import os
import numpy as np
from typing import List, Optional

# Lazy load expensive imports
_faiss = None # No longer used, but keep structure
_embed_model = None
_pd = None # For pandas
_cosine_similarity = None # For scikit-learn

def _lazy_load_pandas():
    global _pd
    if _pd is None:
        try:
            import pandas as pd
            _pd = pd
            print("--- Pandas loaded successfully ---")
        except ImportError:
            raise ImportError("Pandas library not found. Please install it: pip install pandas")
    return _pd
    
def _lazy_load_cosine_similarity():
    global _cosine_similarity
    if _cosine_similarity is None:
        try:
            # Requires scikit-learn
            from sklearn.metrics.pairwise import cosine_similarity
            _cosine_similarity = cosine_similarity
            print("--- Scikit-learn cosine_similarity loaded successfully ---")
        except ImportError:
             raise ImportError("scikit-learn library not found. Please install it: pip install scikit-learn")
    return _cosine_similarity

def _lazy_load_embedding_model():
    global _embed_model
    if _embed_model is None:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            
            _embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
            print(f"--- Google Embedding Model ({_embed_model.model}) loaded ---")
        except ImportError:
            raise ImportError("langchain-google-genai not found. Please install it: pip install langchain-google-genai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Embedding Model: {e}")
    return _embed_model

def embed_product_description(description: Optional[str]) -> Optional[List[float]]:
    """Generates a vector embedding for a single product description string.

    Uses the lazily loaded GoogleGenerativeAIEmbeddings model.

    Args:
        description: The product description text.

    Returns:
        A list of floats representing the embedding, or None if embedding fails 
        or the description is empty.
    """
    if not description:
        print("--- Warning: No product description provided for embedding. ---")
        return None
    
    try:
        model = _lazy_load_embedding_model()
        # Embed single query
        embedding = model.embed_query(description)
        print(f"--- Generated embedding for description (shape: {np.array(embedding).shape}) ---")
        return embedding
    except Exception as e:
        print(f"--- Error embedding product description: {e} ---")
        return None

def embed_product_list(descriptions: List[str]) -> Optional[List[List[float]]]:
    """Generates vector embeddings for a list of product description strings.
    
    Uses the `embed_documents` method of the lazily loaded Google embedding model
    for potential batching efficiency.

    Args:
        descriptions: A list of product description texts.

    Returns:
        A list of embeddings (each a list of floats), or None if embedding fails
        or the input list is empty.
    """
    if not descriptions:
        print("--- Warning: No descriptions provided for batch embedding. ---")
        return None
        
    try:
        model = _lazy_load_embedding_model()
        # Use embed_documents for batching
        embeddings = model.embed_documents(descriptions)
        print(f"--- Generated {len(embeddings)} embeddings for list (shape[0]: {np.array(embeddings[0]).shape}) ---")
        return embeddings
    except Exception as e:
        print(f"--- Error batch embedding product descriptions: {e} ---")
        return None

def search_product_index(query_embedding: Optional[List[float]], index_path: Optional[str], top_k: int = 3, description_column: str = 'Title', rank_column: str = 'Rank') -> Optional[float]:
    """Searches a product CSV index using a query embedding.

    Loads a CSV file specified by `index_path`. Reads product descriptions 
    (from `description_column`) and ranks (from `rank_column`). Embeds the 
    descriptions and calculates cosine similarity between the `query_embedding` 
    and all product description embeddings.

    Args:
        query_embedding: The embedding vector of the item to search for.
        index_path: Path to the CSV file containing product descriptions and ranks.
        top_k: The number of most similar products to consider.
        description_column: The name of the column in the CSV containing product descriptions (or titles).
        rank_column: The name of the column in the CSV containing the pre-calculated rank/score.

    Returns:
        The average rank (float) of the top_k most similar products found in 
        the index, or None if the search fails (e.g., invalid path, missing columns, 
        embedding errors, no valid products found).
    """
    pd = _lazy_load_pandas()
    cosine_similarity = _lazy_load_cosine_similarity()
    
    if query_embedding is None:
        print("--- Cannot search index: No query embedding provided. ---")
        return None
    if not index_path or not os.path.exists(index_path):
        print(f"--- Cannot search index: Index path '{index_path}' is invalid or does not exist. ---")
        return None
        
    try:
        print(f"--- Loading CSV index from: {index_path} ---")
        df = pd.read_csv(index_path)
        
        if description_column not in df.columns:
            print(f"--- Error: Description column '{description_column}' not found in CSV '{index_path}'. Available columns: {list(df.columns)} ---")
            return None
        # Also check for the rank column
        if rank_column not in df.columns:
             print(f"--- Error: Rank column '{rank_column}' not found in CSV '{index_path}'. Available columns: {list(df.columns)} ---")
             return None
            
        # Get descriptions and ranks, handle potential missing values
        descriptions = df[description_column].fillna("").astype(str).tolist()
        # Ensure ranks are numeric, coercing errors to NaN and then potentially filling/dropping
        ranks = pd.to_numeric(df[rank_column], errors='coerce')
        
        # Keep only rows where both description and rank are valid for searching
        valid_indices = ranks.notna().to_numpy().nonzero()[0]
        if len(valid_indices) == 0:
            print(f"--- Warning: No valid descriptions or ranks found in columns '{description_column}'/'{rank_column}' of CSV '{index_path}'. ---")
            return None # Or maybe 0.0?
            
        valid_descriptions = [descriptions[i] for i in valid_indices]
        valid_ranks = ranks.iloc[valid_indices].tolist()
        df_indices_map = {i: original_idx for i, original_idx in enumerate(valid_indices)} # Map from filtered index to original df index
            
        print(f"--- Embedding {len(valid_descriptions)} valid descriptions from CSV... ---")
        index_embeddings = embed_product_list(valid_descriptions)
        
        if index_embeddings is None or len(index_embeddings) == 0:
            print("--- Failed to generate embeddings for index descriptions. Score calculation aborted. ---")
            return None

        # Reshape query embedding for cosine_similarity (expects 2D arrays)
        query_vec = np.array(query_embedding).reshape(1, -1)
        index_vecs = np.array(index_embeddings)

        # Calculate cosine similarities
        # Result shape: (n_query_samples, n_index_samples)
        similarities = cosine_similarity(query_vec, index_vecs)[0] # Get the similarities for the single query
        
        # Get top K indices (indices into the *filtered* list/embeddings)
        actual_k = min(top_k, len(similarities))
        if actual_k <= 0:
             print("--- No similarities calculated or k=0. ---")
             return None # Indicate no rank could be calculated
             
        # Indices of top K similar items within the *filtered* list
        top_k_filtered_indices = np.argsort(similarities)[-actual_k:][::-1]
        top_k_similarities = similarities[top_k_filtered_indices]
        
        # Get the corresponding ranks from the filtered list
        top_k_ranks = [valid_ranks[i] for i in top_k_filtered_indices]
        
        # Aggregate score: Average rank of top_k results
        average_rank = float(np.mean(top_k_ranks))
        
        print(f"--- Index Search Results --- Top {actual_k} Similarities: {top_k_similarities}, Corresponding Ranks: {top_k_ranks}, Average Rank: {average_rank:.2f} ---")
        
        return average_rank
        
    except Exception as e:
        print(f"--- Error searching CSV index '{index_path}': {e} ---")
        # Print traceback for detailed debugging
        import traceback
        traceback.print_exc()
        return None 