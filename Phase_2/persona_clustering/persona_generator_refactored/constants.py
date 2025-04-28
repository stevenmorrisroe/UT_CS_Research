# Constants for the persona generation pipeline
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- File Paths ---
# Get the directory containing this constants.py file
_constants_dir = os.path.dirname(__file__)
# Get the parent directory (persona_generator_refactored)
_script_dir = os.path.dirname(_constants_dir)
# Get the base directory (persona_clustering)
# _base_dir = os.path.dirname(_script_dir) # Old base directory (Phase_2)

# --- File Paths (Modified to be relative to _script_dir, which is persona_clustering) ---
# Default data file path (relative to the script's parent directory)
DEFAULT_DATA_FILE = os.path.join(_script_dir, "data", "amazon-purchases.csv")
# Default output directory (relative to the script's parent directory)
DEFAULT_OUTPUT_DIR = os.path.join(_script_dir, "output_nmf_k20")
# Default survey file path (relative to the script's parent directory)
DEFAULT_SURVEY_FILE = os.path.join(_script_dir, "data", "survey.csv")

# --- Model Parameters ---
N_TOPICS = 20 # Number of personas/topics
MAX_FEATURES = 5000 # Max features for TfidfVectorizer
MIN_DF = 5 # Minimum document frequency for TfidfVectorizer
MAX_DF = 0.90 # Maximum document frequency for TfidfVectorizer
NMF_MAX_ITER = 300 # Max iterations for NMF convergence
NMF_RANDOM_STATE = 42 # Random state for reproducibility
NMF_INIT_METHOD = 'nndsvda' # Initialization method for NMF
NMF_SOLVER = 'mu' # Solver for NMF
NMF_BETA_LOSS = 'kullback-leibler' # Beta loss function for NMF
NMF_L1_RATIO = 0 # L1 ratio for NMF (0 for pure KL divergence with 'kullback-leibler')

# --- Output Configuration ---
N_TOP_WORDS_DISPLAY = 15 # Number of top words to display per topic in logs/outputs
N_WORDS_FOR_PROMPT = 25 # Number of top words/ngrams to use for prompt generation
N_TOP_WORDS_COHERENCE = 10 # Number of top words to use for coherence calculation

# --- Data Cleaning & Filtering ---
MIN_PURCHASE_VALUE = 1.00 # Minimum purchase value to include
MIN_TITLE_LENGTH = 4      # Minimum character length for cleaned titles
EXCLUDE_RAW_CATEGORY_PREFIXES = ('abis_',) # Raw category prefixes to handle/map
EXCLUDE_STD_CATEGORIES = { # Set of standardized category tokens to exclude entirely
    'category_gift_card',
    'category_electronic_gift_card',
    # Removed categories that are now mapped more specifically
}

# --- Category Mapping Constants (Used in text_utils.standardize_category) ---
# Prefixes for specific ABIS category mappings
ABIS_CATEGORY_MAPPINGS = {
    'ABIS_BOOK': 'category_books',
    'ABIS_MUSIC': 'category_music',
    'ABIS_DVD': 'category_movies',
    'ABIS_DOWNLOADABLE_SOFTWARE': 'category_software',
    'ABIS_VIDEO_GAMES': 'category_video_games',
    'ABIS_DRUGSTORE': 'category_health_and_personal_care',
    'ABIS_LAWN_AND_GARDEN': 'category_patio_lawn_and_garden',
    'ABIS_KITCHEN': 'category_home_and_kitchen',
    'ABIS_HOME_IMPROVEMENT': 'category_home_improvement_and_tools',
    'ABIS_HOME': 'category_home_goods',
    'ABIS_SPORTS': 'category_sports_and_outdoors',
    'ABIS_ELECTRONICS': 'category_electronics',
    'ABIS_EBOOKS': 'category_books',
    'ABIS_WIRELESS': 'category_electronics_accessories',
    'ABIS_TOY': 'category_toys_and_games',
    'ABIS_PC': 'category_computers_and_accessories',
    'ABIS_BEAUTY': 'category_beauty_and_personal_care',
    'ABIS_VIDEO': 'category_electronics_tv_and_video',
    'ABIS_PET_PRODUCTS': 'category_pet_supplies',
    # 'ABIS_GIFT_CARD' is excluded via EXCLUDE_STD_CATEGORIES if mapped to category_gift_card
}
# Specific category mappings (raw -> standardized)
SPECIFIC_CATEGORY_MAPPINGS = {
    'skin_moisturizer': 'category_skincare',
    'skin_cleaning_agent': 'category_skincare',
    'skin_cleaning_wipe': 'category_skincare',
    'hair_styling_agent': 'category_haircare',
    'shampoo': 'category_haircare',
    # Add mappings for conditioner etc. if needed
    'oral_hygiene': 'category_oral_care',
    'toothpaste': 'category_oral_care',
    'toothbrush': 'category_oral_care',
    'mouthwash': 'category_oral_care',
    'soap': 'category_bath_and_body',
    'body_wash': 'category_bath_and_body',
    # Add mappings for makeup, cosmetics
    'cellular_phone_case': 'category_phone_accessories',
    'screen_protector': 'category_phone_accessories',
    'portable_electronic_device_cover': 'category_phone_accessories',
    'headphones': 'category_audio_accessories',
    'earbud': 'category_audio_accessories',
    'speaker': 'category_audio_accessories',
    'wearable_computer': 'category_wearable_tech',
    'smart_watch': 'category_wearable_tech',
    'fitness_tracker': 'category_wearable_tech',
    'notebook_computer': 'category_laptops',
    'laptop': 'category_laptops',
    'tablet_computer': 'category_tablets',
    'amazon_tablet': 'category_tablets',
    'physical_video_game_software': 'category_physical_video_games',
    'downloadable_video_game': 'category_digital_video_games',
    'video_game_controller': 'category_gaming_accessories',
    'video_game_accessories': 'category_gaming_accessories',
    'video_game_console': 'category_gaming_consoles',
    'pet_food': 'category_pet_food',
    'pet_toy': 'category_pet_toys',
    'food_storage': 'category_food_storage',
    'computer_drive_or_storage': 'category_computer_storage',
    'security_camera': 'category_security_cameras',
    'camera_tripod': 'category_camera_accessories',
    'camera_other_accessories': 'category_camera_accessories',
}
# Keywords for broader category consolidation (used as fallbacks)
CATEGORY_CONSOLIDATION_KEYWORDS = {
    # More specific first if keywords overlap
    'home_improvement_and_tools': ['tools', 'home_improvement'],
    'home_appliances': ['appliances'],
    'home_organization': ['storage', 'organization'],
    'home_goods': ['bedding', 'bath', 'furniture', 'decor'],
    'home_and_kitchen': ['home', 'kitchen', 'dining'], # More general 'home' checked later
    'beauty_and_personal_care': ['beauty', 'skin', 'hair', 'oral_care', 'personal_care'], # Combined
    'health_supplements': ['vitamin', 'vitamins', 'supplement', 'supplements', 'nutrition'], # Added plurals
    'health_and_medical': ['medical', 'medication'],
    'health_and_personal_care': ['health'], # General health
    'patio_lawn_and_garden': ['outdoor', 'patio', 'garden', 'lawn'],
    'computers_and_accessories': ['computer', 'pc', 'laptop', 'laptops'], # Check before general accessories
    'camera_and_photo': ['camera', 'photo'], # Check before general accessories
    'electronics_audio': ['headphone', 'headphones', 'earbud', 'earbuds', 'speaker', 'speakers', 'audio'],
    'electronics_tv_and_video': ['television', 'video'],
    'electronics_accessories': ['phone', 'mobile', 'wireless_accessory', 'screen_protector', 'portable_electronic_device_cover', 'accessories'], # General accessories checked later
    'electronics': ['electronics'], # General electronics
    'office_products': ['office_product', 'office_supply', 'office_supplies'],
    'clothing': ['clothing', 'shirt', 'pants', 'dress', 'outerwear'],
    'shoes': ['shoes'],
    'jewelry': ['jewelry'],
    'watches': ['watch', 'watches'],
    'luggage_and_bags': ['luggage', 'bag', 'bags'],
    'grocery_pantry': ['pantry'],
    'snacks': ['snack', 'snacks'],
    'beverages': ['beverage', 'beverages', 'coffee', 'tea'],
    'grocery_and_gourmet_food': ['grocery', 'gourmet', 'food'], # General grocery
    'toys_and_games': ['toys', 'game', 'games'],
    'baby_products': ['baby'],
    'pet_supplies': ['pet', 'pets', 'pet_supplies'], # Catch-all for pet items
    'automotive': ['automotive'],
    'sports_and_outdoors': ['sports', 'outdoors'],
    'industrial_and_scientific': ['industrial', 'scientific'],
    'books': ['book', 'books'],
    'music': ['music'],
}


# --- Bigram Configuration ---
BIGRAM_MIN_FREQ = 5 # Minimum frequency for auto-detected bigrams
BIGRAM_TOP_N = 200 # Number of top significant bigrams to consider
WHITELIST_BIGRAMS = { # Manually curated list of important bigrams
    # Common Product Types
    'gift_card', 'video_game', 't_shirt', 'hard_drive', 'cell_phone', 'coffee_maker',
    'light_bulb', 'paper_towels', 'bubble_wrap', 'sd_card', 'usb_c', 'hdmi_cable',
    'power_adapter', 'screen_protector', 'phone_case', 'mouse_pad', 'keyboard_cover',
    'water_bottle', 'travel_mug', 'cutting_board', 'air_fryer', 'slow_cooker',
    'essential_oil', 'bath_bomb', 'face_mask', 'body_wash', 'hand_soap', 'lip_balm',
    'dog_food', 'cat_food', 'cat_litter', 'dog_treats', 'bird_seed',
    'lawn_mower', 'garden_hose', 'flower_pot', 'picture_frame', 'wall_decor',
    'throw_pillow', 'area_rug',

    # Product Attributes/Features
    'gluten_free', 'stainless_steel', 'long_sleeve', 'short_sleeve', 'noise_cancelling',
    'high_definition', 'carbon_fiber', 'king_size', 'queen_size', 'twin_size',
    'extra_large', 'medium_size', 'small_size', 'eco_friendly', 'organic_cotton',
    'water_resistant', 'quick_dry', 'heavy_duty', 'non_stick', 'multi_purpose',

    # Domain Concepts
    'personal_care', 'skin_care', 'hair_care', 'oral_care', 'customer_service',
    'prime_video', 'amazon_basics', 'whole_foods', 'home_improvement', 'arts_crafts',
    'health_beauty', 'sports_outdoors', 'office_supplies', 'pet_supplies', 'baby_product',
    'tv_video',

    # Specific Categories (already somewhat handled, but reinforce)
    'home_kitchen', 'health_personal', 'electronics_accessories',
    'grocery_gourmet', 'toys_games',
}

# --- Stop Words ---
# Combine sklearn's default English stop words with custom e-commerce/generic terms
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    'oz', 'fl', 'ml', 'lb', 'kg', 'g', 'mg', 'count', 'pack', 'pk', 'ct', 'pc', 'pcs',
    'set', 'kit', 'box', 'bag', 'jar', 'can', 'roll', 'tube',
    'size', 'large', 'medium', 'small', 'xl', 'xs', 'x',
    'new', 'free', 'plus', 'extra', 'value', 'assorted', 'variety', 'multi',
    'color', 'black', 'white', 'red', 'blue', 'green', 'yellow', 'silver', 'gold', 'pink', 'purple', 'orange', 'brown', 'gray',
    'mm', 'inch', 'cm', 'ft',
    'men', 'mens', 'women', 'womens', 'kids', 'unisex', # Keep if not part of specific product type
    'use', 'good', 'great', 'best', 'top', 'quality',
    'amazon', 'brand', 'product', 'item',
    'day', 'week', 'month', 'year',
    'shipping', 'delivery',
    'com', 'org', 'net', # From potential URLs/emails in titles
    'ounce', 'inches',
    'asurion', # Specific brand/term to exclude
    'hp', # Specific brand/term to exclude
    'ink', # Specific product term to exclude (too generic here)
    'cartridge', # Specific product term to exclude
    'battery', # Specific product term to exclude
    'alkaline', # Specific product term to exclude
})


# --- Output File Names ---
# Define standard names for output files generated by the pipeline
OUTPUT_FILES = {
    "model": "nmf_model.pkl",
    "vectorizer": "tfidf_vectorizer.pkl",
    "customer_topics": "customer_topic_assignments_nmf.csv",
    "significant_bigrams": "significant_bigrams.txt",
    "category_audit_report": "category_mapping_audit.txt",
    "category_mappings_csv": "category_mappings.csv",
    "persona_prompt_template": "persona_topic_{}_prompt_nmf.txt",
    "persona_top_purchases_template": "persona_topic_{}_top_purchases_nmf.csv",
    "coherence_report": "coherence_report.txt", # File for coherence scores
} 