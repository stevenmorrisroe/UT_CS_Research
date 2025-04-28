import logging
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

# Type variable for structured output models
T = TypeVar('T', bound=BaseModel)

def invoke_structured_llm(model_name: str, temperature: float, prompt: str, output_model: Type[T], model_kwargs: dict | None = None) -> T:
    """Invokes an OpenAI model with structured output parsing."""
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        model_kwargs=model_kwargs or {}
    )
    structured_llm = model.with_structured_output(output_model)
    try:
        # TODO: Add retry logic here using tenacity if needed
        logging.debug(f"Invoking {model_name} with structured output {output_model.__name__}")
        result = structured_llm.invoke(prompt)
        logging.debug(f"Successfully received structured output from {model_name}.")
        return result
    except Exception as e:
        logging.error(f"Error invoking LLM {model_name} for structured output {output_model.__name__}: {e}", exc_info=True)
        # Re-raise the exception to be handled by the calling node/graph logic
        raise e 