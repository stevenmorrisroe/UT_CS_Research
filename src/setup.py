import os
from dotenv import load_dotenv
from pathlib import Path

def load_api_key() -> str:
    """
    Load OpenAI API key from environment or .env file.
    Raises ValueError if API key is not found.
    """
    # Try to load from .env file using absolute path from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(env_path)
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your environment "
            "or .env file."
        )
    
    return api_key

def setup_environment():
    """Setup environment variables for the application."""
    api_key = load_api_key()
    os.environ['OPENAI_API_KEY'] = api_key 
    