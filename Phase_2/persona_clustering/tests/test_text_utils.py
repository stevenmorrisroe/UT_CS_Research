import pytest
import sys
import os

# Adjust path to import from the parent directory's sibling 'persona_generator_refactored'
# This assumes tests are run from the 'persona_clustering' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from persona_generator_refactored.text_utils import clean_text, standardize_category
from persona_generator_refactored.constants import (
    ABIS_CATEGORY_MAPPINGS,
    SPECIFIC_CATEGORY_MAPPINGS,
    CATEGORY_CONSOLIDATION_KEYWORDS
)

# --- Tests for clean_text --- #

@pytest.mark.parametrize("input_text, expected_output", [
    ("Hello World! 123", "hello world "), # Basic cleaning
    ("  Extra   Spaces  ", " extra spaces "),   # Whitespace normalization
    ("Special_Chars&$*", "special_chars"), # Allow underscore
    ("Numbers12345", "numbers"),          # Remove numbers
    (None, ""),                           # None input
    (123, ""),                            # Non-string input
    ("", ""),                              # Empty string
])
def test_clean_text(input_text, expected_output):
    assert clean_text(input_text) == expected_output

# --- Tests for standardize_category --- #

@pytest.mark.parametrize("input_category, expected_output", [
    # None/Empty/Invalid
    (None, None),
    ("", None),
    ("   ", None),
    (123, None),
    # ABIS Mappings
    ("ABIS_BOOK", "category_books"),
    ("ABIS_MUSIC", "category_music"),
    ('ABIS_KITCHEN', 'category_home_and_kitchen'),
    ("ABIS_UNKNOWN", None), # Unmapped ABIS should be None
    ("abis_book", None),    # Check case sensitivity for ABIS keys
    # Specific Mappings (case-insensitive after initial cleaning)
    ("Shampoo", "category_haircare"),
    ("skin_moisturizer", "category_skincare"),
    ("  Headphones & Earbuds ", "category_electronics_audio"),
    ("CELLULAR_PHONE_CASE", "category_phone_accessories"), # Uppercase
    # Keyword Consolidation
    ("Vitamins & Dietary Supplements", "category_health_supplements"),
    ("Baby Toys", "category_toys_and_games"), # Contains 'toys'
    ("Digital Music", "category_music"), # Contains 'music' keyword
    ("Camera Accessories", "category_camera_and_photo"), # Contains 'camera'
    ("Garden Supplies", "category_patio_lawn_and_garden"), # Contains 'garden'
    # Default Prefixing
    ("Some Unique Category", "category_some_unique_category"),
    ("Automotive Parts", "category_automotive"), # Keyword first, then specific
    ("Tools & Home Improvement", "category_home_improvement_and_tools"),
    # Edge Cases with Cleaning
    ("__leading_trailing__", "category_leading_trailing"),
    ("multiple___underscores", "category_multiple_underscores"),
    (" space _ underscore ", "category_space_underscore"),
    ("category with number 123", "category_category_with_number"),
])
def test_standardize_category(input_category, expected_output):
    assert standardize_category(input_category) == expected_output

# Test that all keys in the mapping constants are handled (sanity check)
def test_all_abis_mappings_covered():
    for key in ABIS_CATEGORY_MAPPINGS.keys():
        assert standardize_category(key) == ABIS_CATEGORY_MAPPINGS[key]

def test_all_specific_mappings_covered():
    # Test specific mappings (remembering they are applied after cleaning)
    assert standardize_category("Skin Moisturizer") == SPECIFIC_CATEGORY_MAPPINGS['skin_moisturizer']
    assert standardize_category("cellular phone case") == SPECIFIC_CATEGORY_MAPPINGS['cellular_phone_case']
    assert standardize_category("HEADPHONES") == SPECIFIC_CATEGORY_MAPPINGS['headphones']
    # Add more checks for specific mappings if needed, especially those involving cleaning

def test_keyword_consolidation_examples():
    # Test some examples for keyword consolidation
    assert standardize_category("Vitamin C Supplement") == "category_health_supplements"
    assert standardize_category("Kitchen Utensils") == "category_home_and_kitchen"
    assert standardize_category("Laptop Accessories") == "category_computers_and_accessories" 