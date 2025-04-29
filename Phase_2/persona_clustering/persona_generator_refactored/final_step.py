"""
Processes raw persona data files using Gemini to generate structured buyer persona prompts.
"""

import os
import re
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# --- Configuration ---
INPUT_DIR = "../output_nmf_k20"  # Relative path to the input/output directory
OUTPUT_DIR = "../output_nmf_k20" # Storing results alongside originals
PERSONA_FILE_PATTERN = r"persona_topic_\d+_prompt_nmf\.txt$"
OUTPUT_SUFFIX = "_final_prompt.txt"
GEMINI_MODEL = "gemini-1.5-pro-latest" # Or your preferred Gemini model

# --- Meta Prompt ---
META_PROMPT_TEMPLATE = """
**Meta-Prompt: Generating Buyer Persona Prompts for E-commerce Simulation**

**Objective:** Generate a structured, detailed buyer persona prompt for an LLM agent acting as a buyer in a simulated e-commerce conversation. The persona should be derived *strictly* from the provided data points (keywords, goals, behaviors, demographics) and guide the agent's behavior naturally within the simulation's constraints.

**Input Data:** You will receive a dataset containing:
* Topic Core (Keywords, Inferred Goal)
* Value Sensitivity
* Behavioral Signals (Shopping Style, Item Loyalty)
* Demographics (if available)

```text
{persona_data}
```

**Output Requirements:** Generate a persona prompt adhering to the following structure and guidelines:

**Structure for the Output Persona Prompt:**

1.  **Role:** "You are a potential buyer on an e-commerce platform."
2.  **Persona Name:** [Assign a plausible name based on demographics, if available, otherwise a generic placeholder.]
3.  **Core Goal/Scenario:** [Clearly define the primary reason for this specific interaction, derived from inferred goals and keywords. E.g., "You are looking for a specific type of USB-C cable for your new gaming laptop," or "You need to compare HDMI adapters for your Xbox."]
4.  **Background & Motivation:** [Narrative description synthesizing demographics, core interests (keywords), value sensitivity, and loyalty. Explain *why* they have their goal and *what* drives their preferences. E.g., "You're a gaming enthusiast in your late 20s who values performance. You frequently upgrade your setup and prefer reliable, high-quality cables even if they cost a bit more. You've bought specific brands before and tend to stick with what works."]
5.  **Shopping Style & Behavior:** [Describe their typical online behavior based on engagement, loyalty, and value sensitivity. Crucially, frame this to implicitly guide simulation behavior:]
    * **Imply Focus/Efficiency:** Describe them as goal-oriented, knowing generally what they need, or decisive. (Addresses `Conversation Length`). E.g., "You know what you're generally looking for and prefer to get straight to relevant options," or "You are focused on finding a solution for [Core Goal] efficiently."
    * **Imply Reliance on Seller:** Describe them as needing information from the seller to make a decision. (Addresses `Seller Has Tools`, `No Buyer Tools`). E.g., "You rely on the seller to provide details about product specifications, compatibility, and features to help you choose."
    * **Imply Need for Clarity:** Describe them as needing clear information to make a choice and being direct about their intentions. (Addresses `Be Clear and Direct`, `Goal is Purchase Analysis`). E.g., "You need clear answers about [mention key aspects like compatibility, speed, features based on keywords] before deciding. You will state clearly whether you intend to buy an item once you have the information you need."
    * **Imply Realistic Questions:** Frame their information needs around typical product details. (Addresses `Item Details` limitations). E.g., "Your main questions revolve around standard specifications, compatibility with your devices [mention devices based on keywords like Xbox, laptop], and key features relevant to [mention keywords like gaming, data transfer]."
6.  **Communication Style:** [Briefly describe how they might interact. E.g., "Direct and inquisitive," "Friendly but focused," "Slightly technical."]
7.  **Key Information Needs:** [List 2-3 specific things the persona *must* find out or confirm, based on keywords and goal. E.g., "Confirm USB 3.1 Gen 2 speeds," "Check compatibility with PS5," "Ask about cable length options."]

**Crucial Instructions for the LLM Generating the Persona Prompt:**

* **Strict Data Adherence:** Base the persona *only* on the provided input data. Do not invent details or make assumptions beyond what the data implies.
* **Implicit Constraint Integration:** Weave the behavioral implications of the simulation rules (efficiency, clarity, reliance on seller, realistic questions, decisiveness) into the persona's natural description (motivation, shopping style, communication) as outlined above. **DO NOT explicitly state the simulation rules (like 'You have 10 turns' or 'The seller uses tools') in the generated persona prompt.** The persona's *character* should lead to compliant behavior.
* **Focus on Actionable Guidance:** The final persona prompt must give the buyer LLM agent clear direction on its goals, preferences, and how to interact within the simulated conversation.
* **Narrative Cohesion:** Ensure the different sections of the persona prompt form a consistent and believable character profile.
* **Structure Adherence:** Follow the specified output structure precisely.
"""

# --- Helper Function ---
def generate_final_personas(input_dir: str, output_dir: str, api_key: str | None = None):
    """
    Reads persona data files, generates structured prompts using Gemini, and saves them.

    Args:
        input_dir: Path to the directory containing persona_topic_..._prompt_nmf.txt files.
        output_dir: Path to the directory where final prompts will be saved.
        api_key: Google API key. If None, tries to read from GOOGLE_API_KEY env var.
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("Google API Key not found. Set the GOOGLE_API_KEY environment variable.")
            return

    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
        prompt_template = PromptTemplate.from_template(META_PROMPT_TEMPLATE)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini LLM: {e}")
        return

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    processed_files = 0
    for filename in os.listdir(input_dir):
        if re.match(PERSONA_FILE_PATTERN, filename):
            input_filepath = os.path.join(input_dir, filename)
            output_filename = filename.replace("_prompt_nmf.txt", OUTPUT_SUFFIX)
            output_filepath = os.path.join(output_dir, output_filename)

            # Skip if output file already exists
            if os.path.exists(output_filepath):
                logging.info(f"Skipping '{filename}', output file '{output_filename}' already exists.")
                continue

            logging.info(f"Processing '{filename}'...")
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    persona_data = f.read()

                # Format the prompt using the template and data
                formatted_prompt = prompt_template.format(persona_data=persona_data)

                # Send to Gemini
                message = HumanMessage(content=formatted_prompt)
                response = llm.invoke([message])
                final_persona_prompt = response.content

                # Save the generated prompt
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_persona_prompt)
                logging.info(f"Successfully generated and saved '{output_filename}'")
                processed_files += 1

            except FileNotFoundError:
                logging.warning(f"Input file not found during processing: {input_filepath}")
            except Exception as e:
                logging.error(f"Failed to process '{filename}': {e}")

    logging.info(f"Finished processing. Generated {processed_files} new persona files.")


# --- Main Execution ---
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute paths for input and output directories
    abs_input_dir = os.path.join(script_dir, INPUT_DIR)
    abs_output_dir = os.path.join(script_dir, OUTPUT_DIR)

    # Normalize paths (removes '..')
    abs_input_dir = os.path.normpath(abs_input_dir)
    abs_output_dir = os.path.normpath(abs_output_dir)

    logging.info(f"Starting persona generation process...")
    logging.info(f"Input directory: {abs_input_dir}")
    logging.info(f"Output directory: {abs_output_dir}")

    generate_final_personas(abs_input_dir, abs_output_dir)
    logging.info("Persona generation process complete.") 