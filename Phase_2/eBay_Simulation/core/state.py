from typing import Annotated, Sequence, TypedDict, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SimulationState(TypedDict):
    """State schema for the sales conversation simulation."""
    persona_id: Optional[str]  # Set during initialization via config or default
    current_buyer_prompt: Optional[str]  # Loaded during initialization
    initial_seller_prompt: Optional[str]  # Set during initialization
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history
    product_index_path: Optional[str] = None # Path to the product index for the current persona
    # Sales tracking fields
    sale_completed: bool = False
    sold_item_id: Optional[str] = None
    sold_item_name: Optional[str] = None
    sold_item_price: Optional[float] = None
    sale_timestamp: Optional[datetime] = None
    sale_confidence: Optional[float] = None
    sold_item_description: Optional[str] = None # Extracted description for embedding
    needs_review: bool = False
    sale_details: Optional[str] = None
    product_avg_rank: Optional[float] = None # Semantic similarity score of sold item in persona index 