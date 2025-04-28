# Text cleaning and standardization utilities
import re
import pandas as pd

# Import constants for category mapping
from .constants import (
    EXCLUDE_RAW_CATEGORY_PREFIXES,
    ABIS_CATEGORY_MAPPINGS,
    SPECIFIC_CATEGORY_MAPPINGS,
    CATEGORY_CONSOLIDATION_KEYWORDS
)

def clean_text(text):
    """Simple text cleaning: lowercase, remove non-alphanumeric, remove numbers."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep a-z, spaces, and underscores (often used in category names)
    text = re.sub(r'[^a-z\s_]', '', text) # Allows underscores
    text = re.sub(r'\d+', '', text)      # Remove digits
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    return text

def standardize_category(category):
    """Cleans and standardizes category names for use as tokens."""
    if not isinstance(category, str) or pd.isna(category):
        return None

    category_raw = category # Keep original for ABIS mapping
    category_lower = category.lower()

    # 1. Handle Excluded Prefixes (Mainly ABIS)
    for prefix in EXCLUDE_RAW_CATEGORY_PREFIXES:
        if category_lower.startswith(prefix):
            # Check specific ABIS mappings first
            if category_raw in ABIS_CATEGORY_MAPPINGS:
                return ABIS_CATEGORY_MAPPINGS[category_raw]
            else:
                # If an ABIS category isn't explicitly mapped, exclude it
                return None

    # 2. Basic Cleaning (apply after ABIS check)
    # Replace common separators with underscores
    category = re.sub(r'\s*&\s*|\s*/\s*|\s*-\s*', '_', category_lower)
    # Remove special characters except underscore AND remove digits
    category = re.sub(r'[^a-z\s_]+', '', category)
    category = re.sub(r'\d+', '', category) # Remove digits
    # Replace spaces with underscores and strip leading/trailing underscores
    category = re.sub(r'\s+', '_', category).strip('_')

    # Remove redundant underscores
    category = re.sub(r'_{2,}', '_', category)

    if not category: return None # Return None if cleaning results in empty string

    # 3. Apply Specific Mappings (Exact match on cleaned category)
    if category in SPECIFIC_CATEGORY_MAPPINGS:
        return SPECIFIC_CATEGORY_MAPPINGS[category]

    # 4. Apply Keyword-Based Consolidation (Fallback)
    # Create a space-separated version for robust keyword matching
    category_for_keyword_search = category.replace('_', ' ')
    # Iterate through standardized target categories
    for std_cat, keywords in CATEGORY_CONSOLIDATION_KEYWORDS.items():
        # Check if any keyword is present in the space-separated version
        # Use word boundaries for more precise matching
        if any(re.search(rf'\b{kw}\b', category_for_keyword_search) for kw in keywords):
            # Prepend 'category_' prefix here after finding a match
            return f"category_{std_cat}"

    # 5. Final Check & Default Prefix
    # If no mapping or consolidation rule applied, use the cleaned category name
    # (but only if it wasn't empty or an excluded ABIS category)
    if category: # Ensure it's not an empty string after cleaning
        return f"category_{category}" # Prepend with 'category_'
    else:
        return None # Should be rare, but catches edge cases 