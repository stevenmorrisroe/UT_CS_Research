# Topic coherence calculation functions
import numpy as np
import pandas as pd
import os
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import logging # Add logging import

# Set up logger for this module
logger = logging.getLogger(__name__)

# Import constants
from .constants import N_TOP_WORDS_COHERENCE, OUTPUT_FILES

def calculate_topic_coherence(top_words_per_topic, documents, dtm, feature_names, output_dir, n_top_words=N_TOP_WORDS_COHERENCE):
    """Calculates C_v and UMass topic coherence scores and saves a report.

    Args:
        top_words_per_topic (dict): Map of topic_id to list of top words/ngrams.
        documents (pd.Series): Original text documents used for modeling.
        dtm (scipy.sparse.csr_matrix): Document-Term Matrix (TF-IDF or count).
        feature_names (list): Feature names corresponding to dtm columns.
        output_dir (str): Directory to save the coherence report.
        n_top_words (int): Number of top words per topic for coherence calculation.

    Returns:
        dict: Dictionary containing 'c_v', 'u_mass', and 'u_mass_per_topic' scores.
              Returns scores as None on error.
    """
    logger.info(f"\n--- Calculating Topic Coherence (Top {n_top_words} words) ---")
    coherence_scores = {'c_v': None, 'u_mass': None, 'u_mass_per_topic': None}
    umass_per_topic_details = {}

    # --- C_v Coherence (Gensim) ---
    try:
        logger.info("Preparing data for Gensim C_v coherence...")
        # Simple whitespace tokenization for coherence calculation
        tokenized_docs = [str(doc).split() for doc in documents if pd.notna(doc)]
        if not tokenized_docs:
            logger.error("No valid tokenized documents for Gensim.")
            raise ValueError("No documents for Gensim")

        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

        topics_for_cv = []
        for topic_id, words in top_words_per_topic.items():
            valid_top_words = [word for word in words[:n_top_words] if word in dictionary.token2id]
            if len(valid_top_words) >= 2:
                topics_for_cv.append(valid_top_words)
            else:
                logger.warning(f"Topic {topic_id} has < 2 valid words in Gensim dictionary for C_v. Skipping.")

        if not topics_for_cv:
            logger.error("No topics with sufficient valid words found for C_v calculation.")
        else:
            logger.info(f"Calculating C_v coherence for {len(topics_for_cv)} topics...")
            coherence_model_cv = CoherenceModel(topics=topics_for_cv, texts=tokenized_docs,
                                                corpus=corpus, dictionary=dictionary, coherence='c_v')
            cv_score = coherence_model_cv.get_coherence()
            logger.info(f"Average C_v Coherence: {cv_score:.4f}")
            coherence_scores['c_v'] = cv_score

    except ImportError:
        logger.error("Gensim library not found. Please install (`pip install gensim`) to calculate C_v coherence.")
    except Exception as e:
        logger.error(f"An error occurred during C_v coherence calculation: {e}")

    # --- UMass Coherence --- #
    try:
        logger.info("\nCalculating UMass coherence...")
        epsilon = 1e-12 # Avoid log(0)

        feature_to_index = {name: i for i, name in enumerate(feature_names)}
        dtm_binary = (dtm > 0).astype(int)
        doc_counts_array = np.array(dtm_binary.sum(axis=0)).flatten()
        word_document_counts = {feature_names[i]: doc_counts_array[i]
                                for i in range(len(feature_names))
                                if i < len(doc_counts_array)} # Ensure index exists
        logger.info(f"Calculated document frequencies for {len(word_document_counts)} terms.")

        total_umass_score = 0
        valid_topics_for_umass = 0

        for topic_id, words in top_words_per_topic.items():
            topic_words = words[:n_top_words]
            topic_umass = 0
            word_pairs = 0

            valid_topic_words = [word for word in topic_words if word in feature_to_index]
            if len(valid_topic_words) < 2:
                logger.warning(f"Topic {topic_id} has < 2 valid words in DTM for UMass. Skipping.")
                umass_per_topic_details[topic_id] = None # Mark as not calculated
                continue

            for i, word_i in enumerate(valid_topic_words):
                idx_i = feature_to_index[word_i]
                n_docs_word_i = word_document_counts.get(word_i, 0)

                if n_docs_word_i == 0:
                    continue

                for j in range(i + 1, len(valid_topic_words)):
                    word_j = valid_topic_words[j]
                    idx_j = feature_to_index[word_j]

                    # Calculate co-occurrence efficiently
                    co_occurrence_count = dtm_binary[:, idx_i].multiply(dtm_binary[:, idx_j]).sum()

                    topic_umass += np.log((co_occurrence_count + epsilon) / n_docs_word_i)
                    word_pairs += 1

            if word_pairs > 0:
                avg_topic_umass = topic_umass / word_pairs
                umass_per_topic_details[topic_id] = avg_topic_umass
                total_umass_score += avg_topic_umass
                valid_topics_for_umass += 1
            else:
                logger.warning(f"Topic {topic_id} resulted in 0 valid word pairs for UMass.")
                umass_per_topic_details[topic_id] = None

        if valid_topics_for_umass > 0:
            average_umass = total_umass_score / valid_topics_for_umass
            logger.info(f"Average UMass Coherence (over {valid_topics_for_umass} topics): {average_umass:.4f}")
            coherence_scores['u_mass'] = average_umass
            coherence_scores['u_mass_per_topic'] = umass_per_topic_details
        else:
            logger.error("Could not calculate UMass for any topic.")

    except Exception as e:
        logger.error(f"An error occurred during UMass coherence calculation: {e}")
        # Ensure UMass scores are None if calculation failed
        coherence_scores['u_mass'] = None
        coherence_scores['u_mass_per_topic'] = None

    # --- Save Coherence Report --- #
    report_path = os.path.join(output_dir, OUTPUT_FILES["coherence_report"])
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Topic Coherence Report ===\n\n")
            cv_score_str = f"{coherence_scores['c_v']:.4f}" if coherence_scores['c_v'] is not None else "N/A"
            umass_score_str = f"{coherence_scores['u_mass']:.4f}" if coherence_scores['u_mass'] is not None else "N/A"
            f.write(f"Overall Average C_v Coherence: {cv_score_str}\n")
            f.write(f"Overall Average UMass Coherence: {umass_score_str}\n")

            if coherence_scores['u_mass_per_topic']:
                f.write("\nUMass Coherence per Topic:\n")
                for topic_id, score in sorted(coherence_scores['u_mass_per_topic'].items()):
                    score_str = f"{score:.4f}" if score is not None else "N/A"
                    f.write(f"  - Topic {topic_id}: {score_str}\n")
            f.write(f"\n(Calculated using top {n_top_words} words per topic)\n")
        logger.info(f"Coherence report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error writing coherence report: {e}")

    return coherence_scores 