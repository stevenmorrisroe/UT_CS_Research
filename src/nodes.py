from langchain_openai import ChatOpenAI
from typing import Dict
from src.config import Config
from src.states import InputState, NextStep, Assumptions, Plan
from src.prompts import GENERATE_IDEA
from src.setup import setup_environment
# Ensure API key is set
setup_environment()





def generate_next_idea( State: Plan ,config: Config) -> NextStep:
    """
    Generate next idea using OpenAI API based on input state.
    
    Args:
        config: Configuration object containing model settings
        input_state: Current input state
        
    Returns:
        NextStep object containing the generated idea and assumptions
    """
    input_state = State.input_state[-1]
    model = ChatOpenAI(
        model=config.thinking_model,
        temperature=0
    )
    structured_llm = model.with_structured_output(NextStep)
    # Format the prompt with input state
    formatted_prompt = GENERATE_IDEA.format(
        input_state=input_state.model_dump()
    )
    print(formatted_prompt)
    # Get response from OpenAI
    response = structured_llm.invoke(formatted_prompt)
    
    return response

def


