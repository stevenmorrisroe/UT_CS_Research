# Data loading and preprocessing functions
import pandas as pd
import os
import logging # Add logging import

# Set up logger for this module
logger = logging.getLogger(__name__)

# Import constants
from .constants import (
    MIN_PURCHASE_VALUE,
    MIN_TITLE_LENGTH,
    EXCLUDE_STD_CATEGORIES,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_FILES
)
# Import text utilities
from .text_utils import clean_text, standardize_category

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the Amazon purchase data.

    Args:
        filepath (str): Path to the CSV data file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Aggregated data per customer (Survey_ResponseID, Purchase_Doc, Purchased_ASINs).
            - pd.DataFrame: Original purchase data filtered by value/title length, but before category exclusion.
            Returns (None, None) if loading or processing fails.
    """
    logger.info(f"Loading data from: {filepath}...")
    original_df_filtered = None # Initialize

    try:
        df = pd.read_csv(
            filepath,
            skiprows=1,
            on_bad_lines='warn',
            encoding='utf-8',
            engine='python'
        )
        # Standardize column names (assuming original order)
        df.columns = ['Order_Date', 'Purchase_Price', 'Quantity', 'Shipping_State', 'Title', 'ASIN_ISBN', 'Category', 'Survey_ResponseID']
        logger.info(f"Initial rows loaded: {len(df)}")

    except FileNotFoundError:
        logger.error(f"Data file not found at {filepath}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None, None

    logger.info("Dropping rows with missing essential data (Survey_ResponseID or Title)...")
    initial_rows_count = len(df)
    df.dropna(subset=['Survey_ResponseID', 'Title'], inplace=True)
    logger.info(f"Rows after dropping NAs: {len(df)} (dropped {initial_rows_count - len(df)}) ")

    # --- Convert Price and Quantity --- #
    logger.info("Converting price and quantity to numeric...")
    df['Purchase_Price'] = pd.to_numeric(df['Purchase_Price'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    # Fill NaNs introduced by coercion with 0, assuming non-numeric means zero value/quantity
    # df['Purchase_Price'].fillna(0, inplace=True)
    # df['Quantity'].fillna(0, inplace=True)
    df['Purchase_Price'] = df['Purchase_Price'].fillna(0)
    df['Quantity'] = df['Quantity'].fillna(0)
    df['PurchaseValue'] = df['Purchase_Price'] * df['Quantity']
    logger.info("Calculated PurchaseValue.")

    # --- Filtering Step 1: Value --- #
    initial_rows_count = len(df)
    df = df[df['PurchaseValue'] >= MIN_PURCHASE_VALUE]
    logger.info(f"Rows after filtering by PurchaseValue >= ${MIN_PURCHASE_VALUE:.2f}: {len(df)} (dropped {initial_rows_count - len(df)}) ")

    # --- Clean Titles --- #
    logger.info("Cleaning product titles...")
    df['Clean_Title'] = df['Title'].apply(clean_text)

    # --- Filtering Step 2: Title Length --- #
    initial_rows_count = len(df)
    df = df[df['Clean_Title'].str.len() >= MIN_TITLE_LENGTH]
    logger.info(f"Rows after filtering by Clean_Title length >= {MIN_TITLE_LENGTH}: {len(df)} (dropped {initial_rows_count - len(df)}) ")

    # --- Standardize Categories --- #
    logger.info("Standardizing categories...")
    df['Category_Token'] = df['Category'].apply(standardize_category)

    # Keep a copy of the DF *after* value/title filtering but *before* category exclusion
    # This is used for calculating average purchase value later
    original_df_filtered = df.copy()

    # --- Filtering Step 3: Excluded Categories --- #
    initial_rows_count = len(df)
    # Filter based on the standardized token being in the exclusion list OR being None
    df = df[~df['Category_Token'].isin(EXCLUDE_STD_CATEGORIES)]
    df = df[df['Category_Token'].notna()]
    logger.info(f"Rows after filtering excluded/failed category tokens: {len(df)} (dropped {initial_rows_count - len(df)}) ")

    logger.info("Ensuring ASIN/ISBN are strings...")
    df['ASIN_ISBN'] = df['ASIN_ISBN'].astype(str).fillna('MISSING_ASIN')

    # --- Combine Title and Category Token for Text Input --- #
    # Ensure Category_Token is string before combining (it shouldn't be None here)
    df['Text_Input'] = df['Clean_Title'] + ' ' + df['Category_Token'].astype(str)
    logger.info("Finished combining title and category tokens.")

    # --- Group by Customer --- #
    logger.info("Grouping purchases by customer...")
    # Group text input
    customer_docs = df.groupby('Survey_ResponseID')['Text_Input'].apply(lambda x: ' '.join(x)).reset_index()
    customer_docs.rename(columns={'Text_Input': 'Purchase_Doc'}, inplace=True)
    # Group ASINs (using the filtered df)
    customer_asins = df.groupby('Survey_ResponseID')['ASIN_ISBN'].apply(list).reset_index()
    customer_asins.rename(columns={'ASIN_ISBN': 'Purchased_ASINs'}, inplace=True)
    logger.info("Finished grouping data by customer.")

    # Merge docs and ASINs
    customer_data = pd.merge(customer_docs, customer_asins, on='Survey_ResponseID')

    # --- Final Filtering: Ensure customers have non-empty purchase docs --- #
    initial_cust_count = len(customer_data)
    customer_data = customer_data[customer_data['Purchase_Doc'].str.strip() != '']
    logger.info(f"Final customer count after ensuring non-empty docs: {len(customer_data)} (removed {initial_cust_count - len(customer_data)} customers)")

    logger.info(f"Processed {len(customer_data)} unique customers with filtered data.")
    return customer_data, original_df_filtered


def audit_category_mappings(filepath, output_dir=DEFAULT_OUTPUT_DIR):
    """Analyzes category standardization and saves reports.

    Args:
        filepath (str): Path to the CSV data file.
        output_dir (str): Directory to save audit reports.

    Returns:
        dict or None: Dictionary with audit statistics or None on failure.
    """
    logger.info("\n--- Auditing Category Mappings ---")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(
            filepath,
            skiprows=1,
            on_bad_lines='warn',
            encoding='utf-8',
            engine='python'
        )
        df.columns = ['Order_Date', 'Purchase_Price', 'Quantity', 'Shipping_State', 'Title', 'ASIN_ISBN', 'Category', 'Survey_ResponseID']
    except FileNotFoundError:
        logger.error(f"Data file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading CSV for audit: {e}")
        return None

    initial_rows = len(df)
    df.dropna(subset=['Category'], inplace=True)
    logger.info(f"Starting audit with {len(df)} rows having category data (dropped {initial_rows - len(df)}) ")

    raw_category_counts = df['Category'].value_counts()
    total_raw_categories = len(raw_category_counts)
    logger.info(f"Found {total_raw_categories} unique raw category values")

    df['Standardized_Category'] = df['Category'].apply(standardize_category)

    unmapped_mask = df['Standardized_Category'].isna()
    unmapped_df = df[unmapped_mask]
    mapped_df = df[~unmapped_mask]

    total_rows = len(df)
    unmapped_rows = len(unmapped_df)
    unmapped_pct = (unmapped_rows / total_rows) * 100 if total_rows > 0 else 0

    std_category_counts = mapped_df['Standardized_Category'].value_counts()
    total_std_categories = len(std_category_counts)

    logger.info(f"Standardization results:")
    logger.info(f"  - {unmapped_rows} / {total_rows} ({unmapped_pct:.2f}%) have unmapped categories")
    logger.info(f"  - {total_raw_categories} raw -> {total_std_categories} standardized categories")
    consolidation_ratio = (total_raw_categories / total_std_categories) if total_std_categories > 0 else 0
    logger.info(f"  - Consolidation ratio: {consolidation_ratio:.2f}x")

    unmapped_cat_counts = unmapped_df['Category'].value_counts()
    top_unmapped = unmapped_cat_counts.head(20).to_dict()

    logger.info("\nTop unmapped categories (by frequency):")
    for cat, count in sorted(top_unmapped.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  - '{cat}': {count} occurrences")

    mapping_patterns = {}
    if not mapped_df.empty:
        mapping_patterns = mapped_df.groupby('Standardized_Category')['Category'].apply(lambda x: x.value_counts().to_dict()).to_dict()

    fragmented_categories = []
    for std_cat, raw_cats_dict in mapping_patterns.items():
        if len(raw_cats_dict) > 5:
            fragmented_categories.append({
                'standardized_category': std_cat,
                'raw_count': len(raw_cats_dict),
                'total_occurrences': sum(raw_cats_dict.values()),
                'top_raw_categories': dict(sorted(raw_cats_dict.items(), key=lambda x: x[1], reverse=True)[:5])
            })
    fragmented_categories = sorted(fragmented_categories, key=lambda x: x['raw_count'], reverse=True)

    logger.info(f"\nPotentially fragmented categories (many raw -> one standard):")
    for frag in fragmented_categories[:5]:
        logger.info(f"  - '{frag['standardized_category']}': maps from {frag['raw_count']} raw categories")
        # Ensure top_raw_categories exists and has items before iterating
        if frag.get('top_raw_categories'):
             for raw_cat, count in list(frag['top_raw_categories'].items())[:3]:
                 logger.info(f"      - '{raw_cat}': {count} occurrences")
        else:
             logger.info("      - (No raw categories found)")

    raw_to_std_mapping = {}
    # Optimize: Group by raw category and check unique standardized categories
    if not df.empty:
        grouped_raw = df.groupby('Category')['Standardized_Category'].nunique()
        ambiguous_raw_cats = grouped_raw[grouped_raw > 1].index
        if not ambiguous_raw_cats.empty:
             ambiguous_df = df[df['Category'].isin(ambiguous_raw_cats)]
             raw_to_std_mapping = ambiguous_df.groupby('Category')['Standardized_Category'].value_counts().unstack(fill_value=0).to_dict('index')
             # Convert counts to int if needed
             raw_to_std_mapping = {k: {sk: int(sv) for sk, sv in v.items()} for k, v in raw_to_std_mapping.items()}

    logger.info(f"\nAmbiguous raw categories (one raw -> multiple standard):")
    for raw_cat, std_cats_dict in sorted(raw_to_std_mapping.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
        logger.info(f"  - '{raw_cat}' maps to {len(std_cats_dict)} standardized categories:")
        for std_cat, count in std_cats_dict.items():
             # Handle potential NaN keys from unstack if a category only mapped to NaN
             std_cat_display = 'None/unmapped' if pd.isna(std_cat) else f"'{std_cat}'"
             logger.info(f"      - {std_cat_display}: {count} occurrences")

    # --- Generate recommendations (Simplified suggestion logic) ---
    recommendations = []
    if top_unmapped:
        recommendations.append("Consider adding explicit mappings for these frequent unmapped categories:")
        for cat, count in sorted(top_unmapped.items(), key=lambda x: x[1], reverse=True)[:5]:
            recommendations.append(f"  - '{cat}' ({count} occurrences)")
    if fragmented_categories:
        recommendations.append("\nConsider consolidating these fragmented standardized categories:")
        for frag in fragmented_categories[:3]:
            recommendations.append(f"  - Review '{frag['standardized_category']}' (maps from {frag['raw_count']} raw categories)")
    if raw_to_std_mapping:
        recommendations.append("\nConsider disambiguating these raw categories:")
        for raw_cat in list(raw_to_std_mapping.keys())[:3]:
            recommendations.append(f"  - '{raw_cat}' maps to multiple categories.")
    # --- End Recommendations ---

    results = {
        'total_rows': total_rows,
        'total_raw_categories': total_raw_categories,
        'total_std_categories': total_std_categories,
        'unmapped_rows': unmapped_rows,
        'unmapped_percentage': unmapped_pct,
        'consolidation_ratio': consolidation_ratio,
        'top_unmapped_categories': top_unmapped,
        'fragmented_categories': fragmented_categories,
        'ambiguous_raw_categories': raw_to_std_mapping,
        'recommendations': recommendations
    }

    # --- Save Reports --- #
    report_path = os.path.join(output_dir, OUTPUT_FILES["category_audit_report"])
    mappings_path = os.path.join(output_dir, OUTPUT_FILES["category_mappings_csv"])
    logger.info(f"Saving category audit reports to {output_dir}...")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Category Mapping Audit Report ===\n\n")
            f.write(f"Summary:\n")
            f.write(f"- Rows Analyzed: {total_rows}\n")
            f.write(f"- Raw Categories: {total_raw_categories}\n")
            f.write(f"- Standardized Categories: {total_std_categories}\n")
            f.write(f"- Unmapped Rows: {unmapped_rows} ({unmapped_pct:.2f}%)\n")
            f.write(f"- Consolidation: {consolidation_ratio:.2f}x\n\n")

            f.write(f"Top 20 Unmapped:\n")
            for cat, count in sorted(top_unmapped.items(), key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"- '{cat}': {count}\n")
            f.write("\n")

            f.write(f"Top 10 Fragmented:\n")
            for frag in fragmented_categories[:10]:
                 f.write(f"- '{frag['standardized_category']} ({frag['raw_count']} raw):")
                 if frag.get('top_raw_categories'):
                      top_raw_str = ", ".join([f"'{k}'({v})" for k,v in list(frag['top_raw_categories'].items())[:3]])
                      f.write(f" e.g., {top_raw_str}\n")
                 else:
                      f.write("\n")
            f.write("\n")

            f.write(f"Top 10 Ambiguous:\n")
            for raw_cat, std_cats_dict in sorted(raw_to_std_mapping.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
                 mappings_str = ", ".join([f"{'None' if pd.isna(k) else k}({v})" for k, v in std_cats_dict.items()])
                 f.write(f"- '{raw_cat}' -> {mappings_str}\n")
            f.write("\n")

            f.write("Recommendations:\n")
            f.write("\n".join(recommendations))

        logger.info(f"Audit report saved to {report_path}")

        # Save the full mappings DF to CSV for detailed review
        if not df.empty:
            mappings_df = df[['Category', 'Standardized_Category']].drop_duplicates().sort_values(by=['Standardized_Category', 'Category'])
            mappings_df.to_csv(mappings_path, index=False)
            logger.info(f"Detailed category mappings saved to {mappings_path}")

    except Exception as e:
        logger.error(f"Error writing audit reports: {e}")
        return None # Indicate failure

    logger.info("Category audit finished.")
    return results 