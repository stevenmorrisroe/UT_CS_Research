"""
Utilities for managing buyer personas.

Includes functions to:
- Discover available persona IDs based on files in the data directory.
- Load the specific system prompt associated with a persona ID.
- Create placeholder persona files if none exist.
"""
import os
import re
from typing import List
from pathlib import Path

# Determine Project Root and Persona Directory Path
PROJECT_ROOT = Path(__file__).parents[1]  # Go up two levels to reach project root
#PERSONA_DIR = PROJECT_ROOT / "data" / "personas"  # Old path
#PERSONA_DIR = PROJECT_ROOT / ".." / "persona_clustering" / "output" / "nmf_k20" # Incorrect path from previous attempt
PERSONA_DIR = PROJECT_ROOT / ".." / "persona_clustering" / "output_nmf_k20" # Corrected path

# Placeholder content for the persona file if created
PLACEHOLDER_PERSONA_CONTENT = """
You are a generic online shopper looking for common household items. 
You are price-conscious but open to suggestions. You are generally polite.
"""

def get_available_persona_ids() -> List[str]:
    """Scans the persona data directory for persona files and returns their IDs.
    
    Assumes persona files are named following the pattern 'persona_topic_<id>_prompt_nmf.txt' 
    within the configured PERSONA_DIR.
    
    Returns:
        A list of extracted persona IDs (e.g., ['topic_0', 'topic_1']).
        Returns an empty list if the directory doesn't exist or contains no matching files.
    """
    # Use the configured PERSONA_DIR
    persona_dir = PERSONA_DIR 
    if not persona_dir.is_dir():
        print(f"Warning: Persona directory not found: {persona_dir}")
        return []
        
    persona_ids = [] # Initialize an empty list to store found IDs
    for filename in os.listdir(persona_dir):
        # Match files like 'persona_topic_0_prompt_nmf.txt', 'persona_topic_19_prompt_nmf.txt', etc.
        match = re.match(r"persona_topic_(\d+)_prompt_nmf\.txt", filename) 
        if match:
            persona_ids.append(f"topic_{match.group(1)}") # Add the found ID to the list
            
    # Sort the IDs for consistent ordering (optional but good practice)
    persona_ids.sort(key=lambda x: int(x.split('_')[1])) 
    
    return persona_ids # Return the full list of IDs

def load_persona_prompt(persona_id: str) -> str:
    """Loads the persona system prompt from the corresponding file.

    Constructs the file path based on the configured PERSONA_DIR
    and the provided persona_id (expecting 'persona_topic_<id>_prompt_nmf.txt').

    Args:
        persona_id: The ID of the persona to load (e.g., 'topic_0').

    Returns:
        The content (system prompt) of the persona file as a string.

    Raises:
        FileNotFoundError: If the persona file for the given ID does not exist.
        IOError: If there is an error reading the file.
    """
    # Example: persona_id 'topic_0' -> filename 'persona_topic_0_prompt_nmf.txt'
    topic_number = persona_id.split('_')[1] # Extract the number part
    persona_filename = f"persona_topic_{topic_number}_prompt_nmf.txt"
    # Use the configured PERSONA_DIR
    persona_file = PERSONA_DIR / persona_filename
    
    print(f"--- Attempting to load persona file: {persona_file} ---")
    try:
        with open(persona_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Use the correctly constructed filename in the error message
        raise ValueError(f"Persona prompt file not found: {persona_file}") 
    except Exception as e:
        # Use the correctly constructed filename in the error message
        raise RuntimeError(f"Error loading persona prompt {persona_file}: {e}")

def create_persona_placeholder():
    """Creates a placeholder persona directory and file if none exist.
    
    Checks for the existence of the configured PERSONA_DIR.
    If the directory doesn't exist, it creates it.
    If the directory exists but is empty or contains no '.txt' files,
    it creates a default 'persona_default.txt' file with placeholder content.
    """
    # Use the configured PERSONA_DIR
    persona_dir = PERSONA_DIR
    print(f"--- Checking persona directory: {persona_dir} ---")
    os.makedirs(persona_dir, exist_ok=True)
    
    # Create a README explaining how to obtain or generate persona data
    readme_path = persona_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("""# Persona Data

This directory should contain persona prompt files in the format `persona_topic_X_prompt_nmf.txt`.

The actual persona data files are excluded from version control due to their size.
You can obtain them by either:

1. Running the persona generation scripts (see documentation)
2. Downloading them from the shared data repository
3. Requesting them from the project administrator

## File Format

Each persona file should contain a structured prompt that defines a customer persona derived from NMF analysis of purchase data.
"""
        )
    
    # Create a sample persona file for testing
    sample_path = persona_dir / "persona_topic_0_prompt_nmf.txt"
    with open(sample_path, 'w') as f:
        f.write("""System Prompt: Persona Topic 0

You are a customer persona derived from real purchase data, representing a specific shopping pattern identified using NMF (k=20) on TF-IDF weighted purchase text (including product titles and categories).

Your primary interests, based on keywords like **kitchen, storage, steel, stainless, stainless steel**, suggest you are **primarily shopping within the **home organization** category**. Purchases tend towards **average value items**.

Overall purchasing themes include: kitchen, storage, steel, stainless, stainless steel, organizer, holder, food, category_home_organization, travel, bags, reusable, home, plastic, water, adjustable, silicone, brush, rack, office, mat, cleaning, cleaner, safe, clear.

When responding in a simulated sales conversation:

- Reflect interests aligned with these themes and your primary shopping goal (primarily shopping within the **home organization** category).

- Ask questions or express preferences consistent with someone buying these types of items.

- Your purchase decisions should logically follow from these established interests.

- Do not explicitly state you are a persona; act naturally as this type of customer.
"""
        ) 