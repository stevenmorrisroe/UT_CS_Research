"""
Handles analysis of the conversation to detect sales.

Uses an LLM with structured output (Pydantic) to determine if a sale confirmation 
occurred in the recent message history. Extracts relevant details like item ID, 
name, and price if a sale is detected.
"""
import re
from datetime import datetime
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional, List, Union
from pydantic import BaseModel, Field # Import Pydantic
# Add tenacity imports for retry decorator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import LLM setup from agents (assuming it's safe to share)
# If not, initialize a separate LLM instance here.
from core.agents import get_llm

# --- Pydantic Model for Structured Output ---
class SaleAnalysisOutput(BaseModel):
    """Structured output model for sale analysis."""
    sale_detected: bool = Field(..., description="True if a clear purchase agreement or decision is found, false otherwise.")
    item_id: Optional[str] = Field(None, description="The extracted Item ID (format: v1|...|...), or null if not found/applicable.")
    item_name: Optional[str] = Field(None, description="The extracted Item Name (often from the message preceding the confirmation), or null if not found/applicable.")
    price: Optional[float] = Field(None, description="The extracted price as a number, or null if not found/applicable.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0) based on clarity of intent and detail extraction.")


# --- LLM-based Sale Analyzer ---

# Updated prompt to work with Pydantic structured output
SALE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are an expert conversation analyst. Your task is to analyze the following sequence of messages between a Buyer and a Seller and determine if the Buyer has clearly agreed to purchase a specific item.

Analyze the provided message history and determine:
1.  Did the Buyer explicitly state they want to buy, purchase, order, or checkout a specific item? This includes strong affirmative statements like "I'll take it", "Let's go with that one", "I'll try that one", "Add it to my cart", or similar direct confirmations indicating a decision to purchase.
2.  If yes, can you identify the Item ID (format: v1|...|...), Item Name (often from the message preceding the confirmation), and Price (format: USD X.XX)?

Based on your analysis, populate the fields of the SaleAnalysisOutput structure.

Focus ONLY on clear purchase confirmation or decision in the message history. Distinguish this from general interest, questions about the item, or expressions of potential future interest.
"""),
    # Use MessagesPlaceholder to inject the message history
    MessagesPlaceholder(variable_name="recent_messages")
])

@retry( # Keep retry logic
    # Retry on Pydantic validation errors or general exceptions during the LLM call
    retry=retry_if_exception_type((ValueError, TypeError, KeyError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def analyze_sale_with_llm(recent_messages: List[BaseMessage]) -> Optional[SaleAnalysisOutput]:
    """Uses an LLM with structured output to analyze messages for sale confirmation."""
    try:
        llm = get_llm() # Reuse LLM setup
        # Bind the Pydantic model to the LLM for structured output
        structured_llm = llm.with_structured_output(SaleAnalysisOutput)
        chain = SALE_ANALYSIS_PROMPT | structured_llm

        # Invoke the chain with the messages
        result: SaleAnalysisOutput = chain.invoke({"recent_messages": recent_messages})

        # Pydantic automatically validates the output. If it passes, return the object.
        return result

    except Exception as e:
        # Catch potential Pydantic validation errors or LLM call errors
        print(f"Error during structured LLM sale analysis: {e}")
        return None


# --- Original Regex-based Analysis (kept as fallback/initial check) ---
# Remove the entire regex function
# def analyze_conversation_for_sale_regex(conversation_text: str, messages: List[BaseMessage]) -> Optional[dict]:
#     """Performs the original regex-based sale analysis."""
#     # Indicators of a completed sale
#     purchase_phrases = [
#         r"I('ll| will) buy it",
#         r"I want to (buy|purchase|order)",
#         r"I('ll| will) take it",
#         r"(I'd like|I would like) to (buy|purchase|order)",
#         r"I('m| am) interested in (buying|purchasing|ordering)",
#         r"(buy|purchase|order) (this|that|it) (now|please)",
#         r"add to (cart|basket)",
#         r"check\s?out (now)?",
#         r"place (the |my |an |this )?(order)",
#         r"complete (the |my |this )?(purchase|transaction)",
#         r"ready to (buy|purchase|order)",
#         r"proceed (with|to) (checkout|payment)",
#         r"(confirm|finalize) (the |my |this )?(order|purchase)"
#     ]
    
#     # Item ID extraction pattern
#     item_id_pattern = r"Item ID:?\s*([A-Za-z0-9|]+)"
#     price_pattern = r"USD\s+([\d,.]+)"  # Captures price in USD format
#     item_name_pattern = r"- \*\*Title:\*\* ([^\n]+)"
#     # Pattern to extract description following the specific marker
#     description_pattern = r"\*\*Product Description:\*\*\s*\n(.*)"
    
#     # Check for purchase indications
#     for phrase in purchase_phrases:
#         if re.search(phrase, conversation_text, re.IGNORECASE):
#             print(f"Sale indication found: {phrase}")
            
#             # Try to extract item ID, name and price
#             item_id = None
#             item_match = re.search(item_id_pattern, conversation_text)
#             if item_match:
#                 item_id = item_match.group(1)
#                 print(f"Found item ID: {item_id}")
            
#             item_name = "Unknown Item"
#             name_match = re.search(item_name_pattern, conversation_text)
#             if name_match:
#                 item_name = name_match.group(1).strip()
#                 print(f"Found item name: {item_name}")
            
#             price = None
#             price_match = re.search(price_pattern, conversation_text)
#             if price_match:
#                 try:
#                     price_str = price_match.group(1).replace(",", "")
#                     price = float(price_str)
#                     print(f"Found price: ${price}")
#                 except ValueError:
#                     print(f"Could not convert price '{price_match.group(1)}' to float")
            
#             # Try to extract description
#             description = None
#             # Search in the *full* conversation history (or relevant tool message) for description
#             description_match = None
#             # Also search for the title using the refined pattern in the same messages
#             item_name_match = None # Reset item_name found flag
#             item_name = "Unknown Item" # Reset item_name default
            
#             for msg in reversed(messages): # Look backwards for the most recent tool output
#                 if hasattr(msg, "content") and isinstance(msg.content, str):
#                     # Try finding description
#                     desc_match_in_msg = re.search(description_pattern, msg.content, re.DOTALL | re.IGNORECASE)
#                     if desc_match_in_msg and not description_match: # Find first description match
#                         description = desc_match_in_msg.group(1).strip()
#                         if description == "No description available.":
#                             print("--- Found placeholder description. ---")
#                         else:
#                             print(f"Found description (length: {len(description)})")
#                         description_match = True # Found it
#                     # Try finding title
#                     title_match_in_msg = re.search(item_name_pattern, msg.content)
#                     if title_match_in_msg and not item_name_match: # Find first title match
#                         item_name = title_match_in_msg.group(1).strip()
#                         print(f"Found item name: {item_name}")
#                         item_name_match = True # Found it
#                 # Stop searching if we found both
#                 if description_match and item_name_match:
#                     break
            
#             if not description_match:
#                 print("--- Warning: Could not extract product description. ---")
#             if not item_name_match:
#                 print("--- Warning: Could not extract item name using refined pattern. ---")
            
#             # Compute confidence based on what we could extract
#             confidence = 0.0
#             if item_id:
#                 confidence += 0.3
#             if price:
#                 confidence += 0.3
#             if item_name != "Unknown Item":
#                 confidence += 0.2
            
#             # Sale phrases have some base confidence
#             confidence += 0.2
            
#             # Cap at 1.0
#             confidence = min(confidence, 1.0)
            
#             # Return dict structure similar to LLM if sale detected, otherwise None
#             return {
#                 "sale_detected": True, 
#                 "item_id": item_id,
#                 "item_name": item_name,
#                 "price": price,
#                 "description": description,
#                 "confidence": confidence
#             }
    
#     # No sale detected by regex
#     # print("--- No sale detected by regex analysis ---") # Optional print
#     return None

# --- Main Analysis Function (Combined) ---

def analyze_conversation_for_sale(state) -> dict:
    """Analyzes the conversation history for sale confirmation using an LLM.

    This function is typically called as a node in the LangGraph.
    It uses `analyze_sale_with_llm` to get structured output indicating 
    if a sale was detected and relevant details (item ID, name, price).
    It also attempts to extract the product description using regex from 
    previous messages (often ToolMessages).
    
    Updates the provided state dictionary with sale information if detected.

    Args:
        state (dict): The current LangGraph state dictionary.

    Returns:
        dict: The potentially updated state dictionary.
    """
    print("--- Analyzing Conversation For Sale ---")
    messages = state.get("messages", [])
    
    if len(messages) < 3: # Keep minimum message check
        print("Not enough messages to analyze for sale.")
        return state # Return original state
    
    # Get the recent messages list
    recent_messages = messages[-min(10, len(messages)):] # Use last 10 messages

    llm_result: Optional[SaleAnalysisOutput] = None
    try:
        print("--- Attempting Structured LLM Sale Analysis --- ")
        llm_result = analyze_sale_with_llm(recent_messages)
    except Exception as e:
        print(f"Structured LLM Sale Analysis failed: {e}.")
        # Optionally, handle specific retry errors if needed, but tenacity should handle retries
        
    # --- Update State if Sale Detected by LLM ---
    if llm_result and llm_result.sale_detected:
        print(f"--- LLM detected sale (Confidence: {llm_result.confidence:.2f}) ---")
        # Use the Pydantic model attributes directly
        item_id = llm_result.item_id
        item_name = llm_result.item_name if llm_result.item_name else "Unknown Item"
        price = llm_result.price
        confidence = llm_result.confidence
        description = None # Initialize description

        # Extract description separately using regex (as it might be in tool output)
        # This logic remains useful as description might not be explicit in user text
        description_pattern = r"\*\*Product Description:\*\*\s*\n(.*)"
        item_name_pattern = r"- \*\*Title:\*\* ([^\n]+)" # Also refine name extraction here if needed
        description_match = False
        item_name_refined_match = False
        extracted_item_name = None # Temporary variable for refined name

        for msg in reversed(messages): # Look backwards
            if hasattr(msg, "content") and isinstance(msg.content, str):
                # Extract description
                desc_match_in_msg = re.search(description_pattern, msg.content, re.DOTALL | re.IGNORECASE)
                if desc_match_in_msg and not description_match:
                    description = desc_match_in_msg.group(1).strip()
                    if description == "No description available.":
                        print("--- Found placeholder description. ---")
                    else:
                        print(f"Found description (length: {len(description)}) separately.")
                    description_match = True
                
                # Extract refined item name (prefer this over LLM if found near description)
                name_match_in_msg = re.search(item_name_pattern, msg.content)
                if name_match_in_msg and not item_name_refined_match:
                    extracted_item_name = name_match_in_msg.group(1).strip()
                    print(f"Found refined item name via regex: '{extracted_item_name}'")
                    item_name_refined_match = True

                # Stop if both found
                if description_match and item_name_refined_match:
                    break
        
        # Override LLM item name if a better one was found via regex near description
        if item_name_refined_match and extracted_item_name:
            print(f"--- Overriding LLM item name ('{item_name}') with regex name ('{extracted_item_name}') ---")
            item_name = extracted_item_name
        elif not item_name_refined_match and item_name == "Unknown Item":
             print("--- Warning: Could not extract item name via LLM or Regex. ---")

        if not description_match:
            print("--- Warning: Could not extract product description separately. ---")

        # Construct sale details string
        sale_details = f"Sale detected with confidence {confidence:.2f}. "
        sale_details += f"Item: '{item_name}'. " # Always include name field
        sale_details += f"Price: ${price:.2f}. " if price is not None else "Price: Not found. "
        sale_details += f"Item ID: {item_id}. " if item_id else "Item ID: Not found. "
        sale_details += f"Description found: {'Yes' if description else 'No'}."

        updated_state = state.copy()
        updated_state["sale_completed"] = True
        updated_state["sold_item_id"] = item_id
        updated_state["sold_item_name"] = item_name
        updated_state["sold_item_price"] = price
        updated_state["sale_timestamp"] = datetime.now()
        updated_state["sold_item_description"] = description # Add extracted description
        updated_state["sale_confidence"] = confidence # Use LLM confidence
        updated_state["needs_review"] = confidence < 0.8 # Flag if confidence is low
        updated_state["sale_details"] = sale_details

        print(f"--- Sale Analysis Complete --- {sale_details}")
        return updated_state
    else:
        if llm_result:
            print(f"--- LLM analysis completed, but no sale detected (Confidence: {llm_result.confidence:.2f}) ---")
        else:
            print("--- No sale detected by analysis (LLM analysis might have failed) ---")
        # Return the original state if no sale detected or LLM failed
        return state 