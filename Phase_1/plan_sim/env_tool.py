import os
from dotenv import load_dotenv
from pathlib import Path

def load_api_key() -> str:
    """
    Load OpenAI API key from environment or .env file.
    Raises ValueError if API key is not found.
    """
    # Look for .env file in the project root directory
    project_root = Path(__file__).parent.parent # Go up one level from plan_sim
    env_path = project_root / '.env'
    
    # load_dotenv will not override existing environment variables
    load_dotenv(dotenv_path=env_path)
    
    # Get API key from environment (might have been pre-existing or loaded from .env)
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your environment "
            "or .env file."
        )
    
    return api_key

def setup_environment():
    """Loads API key and ensures it's set in the environment."""
    api_key = load_api_key()
    # Ensure the key is set in os.environ for libraries that expect it
    os.environ['OPENAI_API_KEY'] = api_key 
    