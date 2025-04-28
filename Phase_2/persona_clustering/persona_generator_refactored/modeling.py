# Vectorization, NMF modeling, and topic assignment functions
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import logging # Add logging import

# Set up logger for this module
logger = logging.getLogger(__name__)

# Import constants
from .constants import (
    MAX_DF, MIN_DF, MAX_FEATURES, CUSTOM_STOP_WORDS,
    N_TOPICS, NMF_RANDOM_STATE, NMF_SOLVER, NMF_BETA_LOSS,
    NMF_L1_RATIO, NMF_MAX_ITER, NMF_INIT_METHOD,
    OUTPUT_FILES, N_TOP_WORDS_DISPLAY
)
# Import bigram utilities
from .bigram_utils import create_bigram_tokenizer

def vectorize_text(documents, combined_bigrams_set):
    """Vectorizes text documents using TF-IDF with a custom bigram tokenizer.

    Args:
        documents (pd.Series): Series of text documents (Purchase_Doc).
        combined_bigrams_set (set[str]): Set of significant bigrams to preserve.

    Returns:
        tuple: Contains:
            - scipy.sparse.csr_matrix: Document-Term Matrix (TF-IDF).
            - TfidfVectorizer: Fitted vectorizer object.
            - list: Feature names (vocabulary).
        Returns (None, None, None) if vectorization fails.
    """
    logger.info(f"\nVectorizing text data with TF-IDF (max_features={MAX_FEATURES}, min_df={MIN_DF}, max_df={MAX_DF})...")
    custom_tokenizer = create_bigram_tokenizer(combined_bigrams_set)
    vectorizer = TfidfVectorizer(
        max_df=MAX_DF,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        stop_words=list(CUSTOM_STOP_WORDS),
        tokenizer=custom_tokenizer
    )
    try:
        dtm = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        logger.info(f"Document-Term Matrix shape: {dtm.shape}")
        logger.info(f"Vocabulary size: {len(feature_names)}")

        if dtm.shape[0] == 0 or dtm.shape[1] == 0:
             logger.error("Document-Term Matrix is empty. Check preprocessing and vectorizer settings.")
             return None, None, None

        return dtm, vectorizer, list(feature_names)

    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        return None, None, None

def train_nmf_model(dtm):
    """Trains an NMF model on the Document-Term Matrix.

    Args:
        dtm (scipy.sparse.csr_matrix): Document-Term Matrix (TF-IDF).

    Returns:
        tuple: Contains:
            - NMF: Trained NMF model object.
            - np.ndarray: Document-topic matrix (W).
            - np.ndarray: Topic-term matrix (H).
        Returns (None, None, None) if training fails.
    """
    logger.info(f"\nTraining NMF model (n_components={N_TOPICS})...")
    nmf = NMF(
        n_components=N_TOPICS,
        random_state=NMF_RANDOM_STATE,
        solver=NMF_SOLVER,
        beta_loss=NMF_BETA_LOSS,
        l1_ratio=NMF_L1_RATIO,
        max_iter=NMF_MAX_ITER,
        init=NMF_INIT_METHOD
    )
    try:
        W = nmf.fit_transform(dtm)
        H = nmf.components_
        logger.info("NMF model training complete.")
        logger.info(f"Reconstruction Error (KL Divergence): {nmf.reconstruction_err_:.4f}")
        return nmf, W, H
    except Exception as e:
        logger.error(f"Error during NMF model training: {e}")
        return None, None, None

def save_model_and_vectorizer(nmf_model, vectorizer, output_dir):
    """Saves the trained NMF model.

    Note: The vectorizer is NOT saved due to pickling issues with custom tokenizers.
          It can be recreated using the parameters in constants.py and the vocabulary.

    Args:
        nmf_model (NMF): Trained NMF model.
        vectorizer (TfidfVectorizer): Fitted vectorizer (used to get info, but not saved).
        output_dir (str): Directory to save the model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, OUTPUT_FILES["model"])
    # vectorizer_path = os.path.join(output_dir, OUTPUT_FILES["vectorizer"])
    try:
        joblib.dump(nmf_model, model_path)
        # joblib.dump(vectorizer, vectorizer_path) # Cannot pickle custom tokenizer
        logger.info(f"NMF model saved to {model_path}")
        # print(f"TF-IDF vectorizer saved to {vectorizer_path}")
        logger.info("Vectorizer not saved due to pickling limitations with custom tokenizer.")
    except Exception as e:
        logger.error(f"Error saving NMF model: {e}")

def assign_topics_to_customers(W, customer_data, output_dir):
    """Assigns the most likely topic to each customer based on NMF weights.

    Args:
        W (np.ndarray): Document-topic matrix from NMF.
        customer_data (pd.DataFrame): DataFrame with customer IDs.
        output_dir (str): Directory to save the assignments.

    Returns:
        pd.DataFrame: DataFrame with 'Survey_ResponseID' and assigned 'Topic'.
    """
    logger.info("\nAssigning topics to customers...")
    if W.shape[0] != len(customer_data):
         logger.warning(f"Mismatch between NMF weights ({W.shape[0]}) and customer data ({len(customer_data)}). Assignment might be incorrect.")
         # Attempt assignment anyway, but it might fail or be misaligned

    try:
        # Ensure customer_data index aligns with W rows if possible, otherwise assume direct correspondence
        customer_topics_df = customer_data[['Survey_ResponseID']].copy()
        # Handle potential mismatch by slicing W if necessary and possible
        if W.shape[0] > len(customer_topics_df):
            logger.warning(f"Truncating W matrix rows ({W.shape[0]}) to match customer data ({len(customer_topics_df)}).")
            W_aligned = W[:len(customer_topics_df), :]
        elif W.shape[0] < len(customer_topics_df):
             logger.error(f"Cannot assign topics, W matrix rows ({W.shape[0]}) < customer data ({len(customer_topics_df)}).")
             return None # Cannot proceed with assignment
        else:
             W_aligned = W

        customer_topics_df['Topic'] = np.argmax(W_aligned, axis=1)
        logger.info("Topic assignment complete.")

        # Save assignments
        assignments_path = os.path.join(output_dir, OUTPUT_FILES["customer_topics"])
        os.makedirs(output_dir, exist_ok=True)
        customer_topics_df.to_csv(assignments_path, index=False)
        logger.info(f"Customer topic assignments saved to {assignments_path}")
        return customer_topics_df

    except Exception as e:
        logger.error(f"Error assigning topics or saving assignments: {e}")
        return None

def calculate_topic_value_metrics(customer_topics_df, original_purchase_df):
    """Calculates average purchase value per topic.

    Args:
        customer_topics_df (pd.DataFrame): Customer IDs and their assigned topics.
        original_purchase_df (pd.DataFrame): Original purchase data (filtered for value/title,
                                          containing 'Survey_ResponseID' and 'PurchaseValue').

    Returns:
        dict: Dictionary containing value metrics per topic and overall average.
              Keys: topic_id (int), 'overall_avg_value' (float).
              Values: dict {'AverageValuePerPurchase': float, 'NumberOfPurchases': int, 'TotalValue': float}
    """
    logger.info("\nCalculating purchase value metrics per topic...")
    topic_value_metrics = {}

    if customer_topics_df is None or original_purchase_df is None or 'PurchaseValue' not in original_purchase_df.columns:
        logger.warning("Missing data for value metric calculation. Skipping.")
        return topic_value_metrics

    try:
        # Ensure dtypes match for merging
        original_purchase_df['Survey_ResponseID'] = original_purchase_df['Survey_ResponseID'].astype(str)
        customer_topics_df['Survey_ResponseID'] = customer_topics_df['Survey_ResponseID'].astype(str)


        # Merge topic assignments back to the relevant purchase data
        purchase_df_with_topics = pd.merge(original_purchase_df, customer_topics_df, on='Survey_ResponseID', how='inner')

        if purchase_df_with_topics.empty:
             logger.warning("No matching purchases found after merging topics. Cannot calculate value metrics.")
             return topic_value_metrics

        # Drop rows where merge failed or topic is NaN (shouldn't happen with inner merge)
        purchase_df_with_topics.dropna(subset=['Topic'], inplace=True)
        # Convert Topic to int
        purchase_df_with_topics['Topic'] = purchase_df_with_topics['Topic'].astype(int)

        # Calculate overall average from the *merged* data (ensures same base)
        overall_avg_value = purchase_df_with_topics['PurchaseValue'].mean()
        topic_value_metrics['overall_avg_value'] = overall_avg_value
        logger.info(f"Overall average purchase value (for customers in model): ${overall_avg_value:.2f}")

        # Group by topic and calculate metrics
        grouped_by_topic = purchase_df_with_topics.groupby('Topic')
        for topic_id, group in grouped_by_topic:
            topic_id = int(topic_id)
            total_value = group['PurchaseValue'].sum()
            num_purchases = len(group)
            avg_value_per_purchase = total_value / num_purchases if num_purchases > 0 else 0

            topic_value_metrics[topic_id] = {
                'TotalValue': total_value,
                'NumberOfPurchases': num_purchases,
                'AverageValuePerPurchase': avg_value_per_purchase
            }
            logger.info(f"  Topic {topic_id}: Avg Purchase Value = ${avg_value_per_purchase:.2f} ({num_purchases} purchases)")

    except Exception as e:
        logger.error(f"Error calculating topic value metrics: {e}")
        # Return partially calculated metrics or empty dict

    return topic_value_metrics

def get_top_words_per_topic(model_components, feature_names, n_top_words=N_TOP_WORDS_DISPLAY):
    """Extracts the top words for each topic from the NMF model components.

    Args:
        model_components (np.ndarray): Topic-term matrix (H) from NMF.
        feature_names (list): List of feature names (vocabulary).
        n_top_words (int): Number of top words to retrieve per topic.

    Returns:
        dict: Dictionary mapping topic_id (int) to a list of top words (str).
    """
    top_words = {}
    num_topics = model_components.shape[0]
    logger.info(f"\nExtracting top {n_top_words} words for {num_topics} topics:")
    for topic_idx in range(num_topics):
        if topic_idx >= model_components.shape[0]:
            logger.warning(f"topic_idx {topic_idx} out of bounds for model_components shape {model_components.shape}")
            continue
        topic = model_components[topic_idx, :]
        # Get indices of top words, sorted by weight (descending)
        # Ensure n_top_words doesn't exceed the number of features
        num_features = len(feature_names)
        actual_top_n = min(n_top_words, num_features)
        if actual_top_n <= 0:
            top_feature_indices = []
        else:
            # Use argpartition for efficiency if only top K needed, but argsort is fine for moderate N
            top_feature_indices = topic.argsort()[:-actual_top_n - 1:-1]

        words = [feature_names[i] for i in top_feature_indices if i < num_features] # Double safety check
        top_words[topic_idx] = words
        # Log top words if using the display count
        if n_top_words == N_TOP_WORDS_DISPLAY:
            logger.info(f"Topic {topic_idx}: {', '.join(words)}")
    return top_words
