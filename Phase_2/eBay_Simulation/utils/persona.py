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
    
    Assumes persona files are named following the pattern 'persona_topic_<id>_final_prompt.txt' 
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
        # Match files like 'persona_topic_0_final_prompt.txt', 'persona_topic_19_final_prompt.txt', etc.
        match = re.match(r"persona_topic_(\d+)_final_prompt\.txt", filename) 
        if match:
            persona_ids.append(f"topic_{match.group(1)}") # Add the found ID to the list
            
    # Sort the IDs for consistent ordering (optional but good practice)
    persona_ids.sort(key=lambda x: int(x.split('_')[1])) 
    
    return persona_ids # Return the full list of IDs

def load_persona_prompt(persona_id: str) -> str:
    """Loads the persona system prompt from the corresponding file.

    Constructs the file path based on the configured PERSONA_DIR
    and the provided persona_id (expecting 'persona_topic_<id>_final_prompt.txt').

    Args:
        persona_id: The ID of the persona to load (e.g., 'topic_0').

    Returns:
        The content (system prompt) of the persona file as a string.

    Raises:
        FileNotFoundError: If the persona file for the given ID does not exist.
        IOError: If there is an error reading the file.
    """
    # Example: persona_id 'topic_0' -> filename 'persona_topic_0_final_prompt.txt'
    topic_number = persona_id.split('_')[1] # Extract the number part
    persona_filename = f"persona_topic_{topic_number}_final_prompt.txt"
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

This directory should contain persona prompt files in the format `persona_topic_X_final_prompt.txt`.

The actual persona data files are excluded from version control due to their size.
You can obtain them by either:

1. Running the persona generation scripts (see documentation)
2. Downloading them from the shared data repository
3. Requesting them from the project administrator

## File Format

Each persona file should contain a structured prompt that defines a customer persona derived from NMF analysis of purchase data.
"""
        )
    
    # Create a sample persona file for testing *only if it doesn't exist*
    sample_path = persona_dir / "persona_topic_0_final_prompt.txt"
    if not sample_path.exists():
        print(f"--- Sample persona file {sample_path.name} not found, creating placeholder. ---")
        with open(sample_path, 'w') as f:
            f.write(PLACEHOLDER_PERSONA_CONTENT) # Use the placeholder content
    else:
        print(f"--- Sample persona file {sample_path.name} already exists, skipping placeholder creation. ---")
        
    print(f"--- Placeholder persona directory and sample file check complete at {persona_dir} ---") 