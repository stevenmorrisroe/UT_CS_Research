# Bigram extraction and tokenizer creation
import nltk
import pandas as pd
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import os # For saving bigrams file
import logging # Add logging import

# Set up logger for this module
logger = logging.getLogger(__name__)

# Import constants
from .constants import WHITELIST_BIGRAMS, BIGRAM_MIN_FREQ, BIGRAM_TOP_N, OUTPUT_FILES

# --- NLTK Data Check --- #
def ensure_nltk_data():
    """Downloads required NLTK data if not found."""
    required_data = [('tokenizers/punkt', 'punkt'),
                       ('tokenizers/punkt_tab', 'punkt_tab')] # Added punkt_tab
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
            logger.info(f"NLTK data '{pkg_id}' found.")
        except LookupError:
            logger.info(f"Downloading NLTK '{pkg_id}' tokenizer data...")
            nltk.download(pkg_id, quiet=True)

# Call the check function when the module is imported
ensure_nltk_data()
# --- End NLTK Data Check --- #

def find_significant_bigrams(documents, output_dir, min_freq=BIGRAM_MIN_FREQ, top_n=BIGRAM_TOP_N):
    """Identifies statistically significant bigrams and combines with whitelist.

    Args:
        documents (pd.Series): Collection of text documents (Purchase_Doc).
        output_dir (str): Directory to save the significant bigrams file.
        min_freq (int): Minimum frequency for a bigram to be considered.
        top_n (int): Number of top bigrams to return based on PMI score.

    Returns:
        set[str]: A set of combined significant and whitelisted bigrams (e.g., {'video_game'}).
    """
    logger.info(f"--- Finding Significant Bigrams (min_freq={min_freq}, top_n={top_n}) ---")
    if documents.empty:
        logger.warning("Input documents series is empty. Using only whitelist bigrams.")
        return WHITELIST_BIGRAMS

    auto_bigrams_list = []
    try:
        logger.info("Tokenizing documents for bigram analysis...")
        # Simple whitespace split for collocation finder
        tokenized_docs = [str(doc).split() for doc in documents if pd.notna(doc)]

        if not tokenized_docs:
            logger.warning("No valid documents found after tokenization. Using only whitelist bigrams.")
            return WHITELIST_BIGRAMS

        logger.info(f"Processing {len(tokenized_docs)} documents for collocations.")
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_documents(tokenized_docs)

        logger.info(f"Applying frequency filter (min_freq={min_freq})...")
        finder.apply_freq_filter(min_freq)

        logger.info(f"Finding top {top_n} bigrams based on PMI score...")
        scored_bigrams = finder.score_ngrams(bigram_measures.pmi)
        top_bigrams_with_scores = scored_bigrams[:top_n]
        auto_bigrams_list = [bigram for bigram, score in top_bigrams_with_scores]

        logger.info(f"Found {len(auto_bigrams_list)} significant bigrams meeting the criteria.")
        if auto_bigrams_list:
            logger.info("Top 5 significant bigrams (by PMI):")
            for i, (bigram, score) in enumerate(top_bigrams_with_scores[:5]):
                logger.info(f"  {i+1}. {' '.join(bigram)} (PMI: {score:.2f})")

            # Save significant bigrams to file
            os.makedirs(output_dir, exist_ok=True)
            bigrams_filename = os.path.join(output_dir, OUTPUT_FILES["significant_bigrams"])
            try:
                with open(bigrams_filename, 'w', encoding='utf-8') as f:
                    for w1, w2 in auto_bigrams_list:
                        f.write(f"{w1}_{w2}\n") # Save in underscore format
                logger.info(f"Auto-detected significant bigrams saved to {bigrams_filename}")
            except Exception as e:
                logger.error(f"Error writing significant bigrams file: {e}")

    except ImportError:
        logger.error("NLTK library not found or data missing. Cannot extract bigrams automatically. Using only whitelist.")
        # Fallback to just whitelist if NLTK fails
        return WHITELIST_BIGRAMS
    except Exception as e:
        logger.error(f"Error during significant bigram extraction: {e}. Using only whitelist bigrams.")
        return WHITELIST_BIGRAMS

    # Format auto-detected bigrams with underscore
    significant_bigrams_set_auto = {f"{w1}_{w2}" for w1, w2 in auto_bigrams_list}

    # Combine with whitelist
    combined_bigrams_set = significant_bigrams_set_auto.union(WHITELIST_BIGRAMS)
    logger.info(f"Found {len(significant_bigrams_set_auto)} auto bigrams, {len(WHITELIST_BIGRAMS)} whitelisted. Total combined: {len(combined_bigrams_set)}")

    return combined_bigrams_set

def create_bigram_tokenizer(significant_bigrams_set):
    """Factory function to create a custom tokenizer that preserves significant bigrams.

    Args:
        significant_bigrams_set (set[str]): Set of bigrams joined by underscore.

    Returns:
        callable: A tokenizer function for scikit-learn vectorizers.
    """
    def tokenizer(text):
        tokens = str(text).split() # Simple split, assumes pre-cleaned text
        if not tokens:
            return []

        processed_tokens = []
        i = 0
        while i < len(tokens):
            # Check for bigram at current position
            if i < len(tokens) - 1:
                potential_bigram = f"{tokens[i]}_{tokens[i+1]}"
                if potential_bigram in significant_bigrams_set:
                    processed_tokens.append(potential_bigram)
                    i += 2 # Skip next token
                    continue

            # If not a significant bigram, add the single token
            processed_tokens.append(tokens[i])
            i += 1

        return processed_tokens

    return tokenizer 