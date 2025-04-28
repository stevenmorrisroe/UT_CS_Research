# Main execution script for NMF Persona Generation Pipeline
import argparse
import sys
import os
import pandas as pd # Add pandas import
import logging # Add logging import

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to stdout

# Add the parent directory to the path to allow relative imports
# (Assuming main.py is run from within persona_generator_refactored directory)
# If run from parent directory, this might not be needed depending on structure
# Consider using a more robust package structure if this becomes complex
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Import constants and utility functions from sibling modules
from .constants import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR, N_TOPICS, N_WORDS_FOR_PROMPT, DEFAULT_SURVEY_FILE
from .data_processing import load_and_preprocess_data, audit_category_mappings
from .bigram_utils import find_significant_bigrams
from .modeling import (
    vectorize_text,
    train_nmf_model,
    save_model_and_vectorizer,
    assign_topics_to_customers,
    calculate_topic_value_metrics,
    get_top_words_per_topic
)
from .coherence_utils import calculate_topic_coherence
from .persona_utils import (
    generate_persona_prompt,
    aggregate_top_purchases_by_frequency,
    save_personas_and_purchases,
    extract_behavioral_signals,
    aggregate_demographics_for_topic # Add new import
)

def main():
    """Main function to run the NMF persona generation pipeline."""
    parser = argparse.ArgumentParser(description='NMF Topic Modeling for Customer Purchase Data')
    parser.add_argument('--audit-only', action='store_true',
                        help='Run only the category mapping audit, skip model training')
    parser.add_argument('--full', action='store_true',
                        help='Run both category audit and full model training')
    parser.add_argument('--data-file', type=str, default=DEFAULT_DATA_FILE,
                        help=f'Path to purchase data file (default based on script location)')
    # Add argument for survey data file - adjusted default and help
    parser.add_argument('--survey-file', type=str, default=DEFAULT_SURVEY_FILE,
                        help='Path to survey data file containing demographics (default based on script location)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory for output files (default based on script location)')

    args = parser.parse_args()

    data_file = args.data_file
    survey_file = args.survey_file # Get survey file path
    output_dir = args.output_dir

    logging.info(f"--- Persona Generation Pipeline --- ")
    logging.info(f"Purchase Data File: {data_file}")
    logging.info(f"Survey Data File: {survey_file}") # Print survey file path
    logging.info(f"Output Directory: {output_dir}")
    mode = 'Audit Only' if args.audit_only else ('Full Run' if args.full else 'Model Training Only')
    logging.info(f"Mode: {mode}")
    logging.info("-----------------------------------")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Demographic Data ---
    demographics_df = None
    if os.path.exists(survey_file):
        try:
            demographics_df = pd.read_csv(survey_file)
            # Basic validation - check for Survey ResponseID
            if 'Survey ResponseID' not in demographics_df.columns:
                logging.warning(f"'Survey ResponseID' column not found in {survey_file}. Cannot link demographics.")
                demographics_df = None
            else:
                 logging.info(f"Successfully loaded demographic data from {survey_file}.")
                 # Rename columns for easier use (optional, but recommended)
                 demographics_df.rename(columns={
                     'Survey ResponseID': 'Survey_ResponseID',
                     'Q-demos-age': 'Age',
                     'Q-demos-hispanic': 'Hispanic',
                     'Q-demos-race': 'Race',
                     'Q-demos-education': 'Education',
                     'Q-demos-income': 'Income',
                     'Q-demos-gender': 'Gender',
                     'Q-sexual-orientation': 'SexualOrientation',
                     'Q-demos-state': 'State'
                     # Add others if needed
                 }, inplace=True)

        except Exception as e:
            logging.warning(f"Failed to load or process demographic data from {survey_file}. Error: {e}. Proceeding without demographics.")
            demographics_df = None
    else:
        logging.warning(f"Survey file not found at {survey_file}. Proceeding without demographics.")
    # --- End Load Demographic Data ---

    # 1. Category Audit (Optional)
    if args.audit_only or args.full:
        logging.info("\nRunning category mapping audit...")
        audit_results = audit_category_mappings(data_file, output_dir)
        if not audit_results:
            logging.error("Category audit failed. Check data file path and format.")
            if args.audit_only:
                sys.exit(1) # Exit if only auditing and it failed
        elif args.audit_only:
            logging.info("\nCategory audit completed successfully. Exiting as --audit-only was specified.")
            sys.exit(0)
        # If --full, continue to model training after audit
        logging.info("\nCategory audit finished. Proceeding with model training...")

    if args.audit_only: # Should have exited above, but as a safeguard
        return

    # 2. Load and Preprocess Data
    customer_data, original_purchase_df = load_and_preprocess_data(data_file)
    if customer_data is None or customer_data.empty:
        logging.error("\nError: Failed to load or preprocess data. Exiting.")
        sys.exit(1)
    logging.info("\nData loaded and preprocessed successfully.")
    documents = customer_data['Purchase_Doc']
    # Create a map for easy ASIN lookup later
    customer_asin_map = customer_data.set_index('Survey_ResponseID')['Purchased_ASINs'].to_dict()

    # 3. Find Significant Bigrams
    combined_bigrams = find_significant_bigrams(documents, output_dir)

    # 4. Vectorize Text
    dtm, vectorizer, feature_names = vectorize_text(documents, combined_bigrams)
    if dtm is None:
        logging.error("\nError: Text vectorization failed. Exiting.")
        sys.exit(1)

    # 5. Train NMF Model
    nmf_model, W, H = train_nmf_model(dtm)
    if nmf_model is None:
        logging.error("\nError: NMF model training failed. Exiting.")
        sys.exit(1)

    # 6. Save Model and Vectorizer
    save_model_and_vectorizer(nmf_model, vectorizer, output_dir)

    # 7. Assign Topics to Customers
    customer_topics_df = assign_topics_to_customers(W, customer_data, output_dir)
    if customer_topics_df is None:
        logging.error("\nError: Failed to assign topics to customers. Exiting.")
        # Continue without value metrics or aggregated purchases if assignment fails?
        # For now, exit, as downstream steps depend on it.
        sys.exit(1)

    # 8. Calculate Topic Value Metrics
    topic_value_metrics = calculate_topic_value_metrics(customer_topics_df, original_purchase_df)

    # 9. Get Top Words (get more for prompts, display fewer)
    top_words_for_prompts = get_top_words_per_topic(H, feature_names, n_top_words=N_WORDS_FOR_PROMPT)
    # Note: get_top_words_per_topic also prints the top N_TOP_WORDS_DISPLAY words

    # 10. Calculate Topic Coherence (using standard number of words)
    # We need the top words specifically for coherence calc (e.g., top 10)
    top_words_for_coherence = get_top_words_per_topic(H, feature_names, n_top_words=10) # Or use N_TOP_WORDS_COHERENCE constant
    coherence_results = calculate_topic_coherence(
        top_words_per_topic=top_words_for_coherence,
        documents=documents,
        dtm=dtm,
        feature_names=feature_names,
        output_dir=output_dir
    )
    logging.info(f"Coherence Calculation Results: {coherence_results}")

    # 11. Generate Personas & Extract Behavioral/Demographic Signals
    logging.info("\nGenerating persona prompts and extracting signals...")
    personas = {}
    all_behavioral_insights = {} # Optional: Store insights if needed elsewhere
    all_demographic_insights = {} # Store demographic insights

    # Check if necessary dataframes are available
    if customer_topics_df is None:
        logging.warning("Skipping persona generation and signal analysis due to missing customer topic assignments.")
    # elif original_purchase_df is None: # Original logic handled missing purchase data
    #     logging.warning("Skipping behavioral analysis due to missing original purchase data. Prompts will lack behavioral details.")
    #     # ... (existing logic for no behavioral data) ...
    else:
        # Proceed with both persona generation and signal extraction
        for topic_id in range(N_TOPICS):
            # Extract behavioral signals (if purchase data available)
            if original_purchase_df is not None:
                behavioral_insights = extract_behavioral_signals(
                    topic_id,
                    original_purchase_df, # Needs original df with date/ASIN
                    customer_topics_df
                )
                all_behavioral_insights[topic_id] = behavioral_insights # Store if needed
                logging.info(f"  Extracted behavioral insights for Topic {topic_id}")
            else:
                behavioral_insights = {} # Pass empty dict if no purchase data
                logging.warning(f"  Skipping behavioral insights for Topic {topic_id} (missing purchase data)")

            # Extract demographic signals (if demographic data available)
            if demographics_df is not None:
                 demographic_insights = aggregate_demographics_for_topic(
                     topic_id,
                     customer_topics_df,
                     demographics_df
                 )
                 all_demographic_insights[topic_id] = demographic_insights # Store if needed
                 logging.info(f"  Extracted demographic insights for Topic {topic_id}")
            else:
                 demographic_insights = {} # Pass empty dict if no demographic data
                 logging.warning(f"  Skipping demographic insights for Topic {topic_id} (missing survey data)")

            # Generate persona prompt including available signals
            personas[topic_id] = generate_persona_prompt(
                topic_id,
                top_words_for_prompts, # Use the more extensive list for prompt context
                topic_value_metrics,
                behavioral_insights, # Pass extracted behavioral insights (or {} if unavailable)
                demographic_insights # Pass extracted demographic insights (or {} if unavailable)
            )
            logging.info(f"Generated prompt for Topic {topic_id} (snippet):")
            logging.info(personas[topic_id][:200] + "...")

    # 12. Aggregate Purchases per Topic
    # Ensure customer_topics_df exists before aggregating
    if customer_topics_df is not None:
        # aggregated_purchases = aggregate_purchases_for_topics(customer_topics_df, customer_asin_map) # Old call
        # New call using the original purchase df which contains titles
        aggregated_top_purchases = aggregate_top_purchases_by_frequency(
            customer_topics_df,
            original_purchase_df, # Pass the full purchase data
            top_n=100 # Specify top N, default is 100 in the function
        )
    else:
        logging.warning("Skipping purchase aggregation due to missing customer topic assignments.")
        # Create empty structure matching the new expected output (dict of empty DFs)
        aggregated_top_purchases = {i: pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency']) for i in range(N_TOPICS)}

    # 13. Save Personas and Purchases
    # Ensure personas dict is populated even if generation was skipped/partial
    if not personas:
         logging.warning("No personas were generated to save.")
    else:
        # Pass the new aggregated_top_purchases dictionary
        save_personas_and_purchases(personas, aggregated_top_purchases, output_dir)

    logging.info(f"\n--- Pipeline Finished Successfully --- ")
    logging.info(f"All outputs saved in '{output_dir}'.")
    logging.info("------------------------------------")

if __name__ == "__main__":
    main() 