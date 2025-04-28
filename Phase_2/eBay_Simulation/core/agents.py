"""
Defines the Seller and Buyer agent nodes for the LangGraph simulation.

Includes:
- LLM initialization (`get_llm`) with retry logic.
- Tool definitions (`@tool`) for eBay API interaction (search, item details).
- System prompts for the Seller agent.
- Agent node functions (`seller_agent_node`, `buyer_agent_node`) that invoke 
  the LLM with appropriate prompts, tools (for seller), and conversation history.
- Error handling and fallback mechanisms for agent invocations.
"""
import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from tools.ebay_api import search_ebay, answer_item_question
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- LLM Setup ---

def get_llm():
    """Initializes and returns the primary ChatGoogleGenerativeAI LLM instance.
    
    Reads the GOOGLE_API_KEY from environment variables.
    Configures the model name (currently hardcoded), temperature, timeout, and retries.
    Includes a retry mechanism for initialization errors.

    Raises:
        ValueError: If GOOGLE_API_KEY is not found in the environment.
        RuntimeError: If the LLM fails to initialize after retries.
        
    Returns:
        An initialized ChatGoogleGenerativeAI instance.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("ERROR: GOOGLE_API_KEY not found in environment variables.")

    print(f"Using API key (first 10 chars): {google_api_key[:10]}...")
    
    # Add retry wrapper for API calls
    @retry(
        retry=retry_if_exception_type((ValueError, TypeError, KeyError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def create_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001", 
            temperature=0.7, 
            google_api_key=google_api_key,
            timeout=60,  # Corrected parameter name
            max_retries=2,  # Built-in retry mechanism
        )
        
    return create_llm()

# --- Tool Definitions ---

@tool
def ebay_search_tool(
    query: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> str:
    """Searches eBay for items matching the query, optionally filtering by price.

    Use this tool to find product information like price, availability, and item IDs on eBay.
    You MUST provide the Item IDs found in the results to the user, as they are needed
    for the 'answer_item_question_tool'.

    Args:
        query: The search term for the product.
        min_price: Optional minimum price filter.
        max_price: Optional maximum price filter.

    Returns:
        A string summarizing the search results found on eBay, including Item IDs.
    """
    print(f"--- Calling eBay Search Tool --- Query: {query}, Min: {min_price}, Max: {max_price}")
    results = search_ebay(query=query, min_price=min_price, max_price=max_price)
    print(f"--- Search Tool Results ---\n{results}")
    return results

@tool
def answer_item_question_tool(item_id: str) -> str:
    """
    Retrieves a standardized **summary** of details for a specific eBay item using its Item ID.
    Use this ONLY when the buyer asks for more details about a specific item identified by its ID,
    or when you want to provide a summary after presenting search results.
    
    **IMPORTANT:** This tool provides a general summary (Title, Price, Condition, Description, etc.).
    It does NOT directly answer specific questions like 'Does it come in blue?' or 'What is the warranty?'.
    If the buyer asks a specific question not covered by the summary, state that the information isn't available in the summary.

    Args:
        item_id: The eBay Item ID (obtained from 'ebay_search_tool' results).

    Returns:
        A markdown formatted string summarizing the item's key details, including a clearly marked 'Product Description:' section.
    """
    print(f"--- Calling Answer Item Question Tool --- Item ID: {item_id}")
    # Simulate getting details - in a real scenario, this would call the ebay_api.answer_item_question
    # We modify the *return format* here to include the description explicitly.
    results = answer_item_question(item_id=item_id) # Assume this function returns a dict or object
    
    # Construct the formatted string output, ensuring description is present
    if isinstance(results, dict):
        formatted_output = f"""**Item Summary (ID: {item_id})**
        
- Title: {results.get('title', 'N/A')}
- Price: {results.get('price', 'N/A')}
- Condition: {results.get('condition', 'N/A')}
- Seller: {results.get('seller', 'N/A')}
- Shipping: {results.get('shipping_cost', 'N/A')}
- Returns: {results.get('returns', 'N/A')}
- Link: {results.get('view_item_url', 'N/A')}

**Product Description:**
{results.get('description', 'No description available.')} 
        """
    else: # Handle case where results might be a simple string (e.g., error message)
        formatted_output = str(results)
        # Ensure description placeholder if it's just an error string
        if "Product Description:" not in formatted_output:
             formatted_output += "\n\n**Product Description:**\nNo description available."
             
    print(f"--- Answer Tool Results (Formatted) ---\n{formatted_output}")
    return formatted_output

# List of tools available to the seller
seller_tools = [ebay_search_tool, answer_item_question_tool]

# --- Default Prompts ---

DEFAULT_SELLER_SYSTEM_PROMPT = """
You are a helpful and knowledgeable eBay sales assistant.
Your primary goal is to understand the customer's needs, help them find suitable products using your tools, and guide them towards a purchase decision if appropriate.

**Conversation Strategy:**
1.  **Engage & Understand:** Greet the customer warmly. Ask open-ended and clarifying questions to fully understand what they are looking for *before* rushing to search. Consider their implied persona from their messages.
2.  **Strategic Tool Use:** You have access to two tools:
    *   `ebay_search_tool`: Use this to find relevant eBay listings based on the customer's refined needs. Always present the Item IDs from the results, as they are needed for the other tool.
    *   `answer_item_question_tool`: Use this to retrieve a **summary** of details for a *specific* item using its Item ID. Use this when the customer expresses interest in a specific item or asks for more general information about it. **Remember:** This tool provides a summary only; if the customer asks a specific question not in the summary, acknowledge that.
3.  **Avoid Redundant Tool Calls:** Don't use `answer_item_question_tool` unless the customer asks about a specific item or you are providing requested details. Don't call search repeatedly without new input from the customer.
4.  **Direct Responses:** Respond directly (without tools) for greetings, asking clarifying questions, summarizing previous findings, checking for understanding, answering simple questions not requiring item details, and managing the conversation flow (e.g., "Does that sound right?", "Is there anything else I can help you find?").
5.  **Guide Towards Decision:** Periodically check if the customer has found what they need or if they require further assistance. If they seem satisfied with an item, you can ask if they are ready to purchase or if they have remaining questions.
6.  **Be Helpful & Professional:** Maintain a friendly, efficient, and professional tone throughout the conversation.

**Output Format:** Your response must be **EITHER** a direct message to the customer **OR** a call to **ONE** tool. Never both.
"""

# --- Agent Nodes ---

def seller_agent_node(state):
    """Represents the Seller agent's turn in the LangGraph.
    
    Invokes the LLM with the seller's system prompt, conversation history,
    and bound tools (eBay search, item details).
    Handles potential errors during LLM invocation with a fallback mechanism.

    Args:
        state (SimulationState): The current state of the simulation graph.

    Returns:
        dict: A dictionary containing the AIMessage generated by the seller 
              (potentially including tool calls) to be added to the state's 
              'messages' list.
    """
    print("--- Seller Node ---")
    # Check if prompts are initialized, handle potential race condition if init fails
    initial_seller_prompt = state.get("initial_seller_prompt")
    if not initial_seller_prompt:
        print("Error: initial_seller_prompt not found in state. Initialization might have failed.")
        return {"messages": [AIMessage(content="Error: Seller configuration missing.")]}

    # Get the LLM
    llm = get_llm()
    # Bind tools to the LLM for the seller agent
    seller_llm_with_tools = llm.bind_tools(seller_tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", initial_seller_prompt),  # Use the prompt set in state
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    seller_runnable = prompt | seller_llm_with_tools
    
    # Check that messages exist and are not empty
    messages = state.get("messages", [])
    if not messages:
        print("Warning: No messages found in state. Adding a default message.")
        messages = [HumanMessage(content="Hello, I'd like some assistance.")]
    
    try:
        response = seller_runnable.invoke({"messages": messages})
        # Ensure response is always AIMessage
        print(f"--- Seller Response --- Type: {type(response)}")
        if response.content:
             print(f"Content: {response.content[:100]}...")
        if hasattr(response, 'tool_calls') and response.tool_calls:
             print(f"Tool Calls: {response.tool_calls}")
        # The response (AIMessage with or without tool_calls) is added to the state messages
        return {"messages": [response]}
    except Exception as e:
        print(f"Error in seller_node: {e}")
        error_message = AIMessage(content=f"Apologies, I encountered an error: {str(e)[:100]}... Let me try a simpler response.")
        # Attempt a simpler response without tools as fallback
        try:
            print("Attempting fallback response without tools...")
            simple_llm = get_llm()  # Get a fresh LLM instance
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful salesperson. Respond briefly and professionally."),
                MessagesPlaceholder(variable_name="messages")
            ])
            simple_runnable = simple_prompt | simple_llm
            fallback_response = simple_runnable.invoke({"messages": messages[-3:] if len(messages) > 3 else messages})
            print(f"Generated fallback response: {fallback_response.content[:100]}...")
            return {"messages": [fallback_response]}
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return {"messages": [error_message]}


def buyer_agent_node(state):
    """Represents the Buyer agent's turn in the LangGraph.

    Invokes the LLM using the specific buyer persona prompt loaded into the state
    and the current conversation history. The buyer agent does not have tools.
    Handles potential errors during LLM invocation with a fallback mechanism.

    Args:
        state (SimulationState): The current state of the simulation graph.

    Returns:
        dict: A dictionary containing the HumanMessage generated by the buyer 
              to be added to the state's 'messages' list.
    """
    print("--- Buyer Node ---")
    # Check if prompts are initialized
    current_buyer_prompt = state.get("current_buyer_prompt")
    if not current_buyer_prompt:
         print("Error: current_buyer_prompt not found in state. Initialization might have failed.")
         # Return message and let the graph proceed (seller will likely error or end)
         return {"messages": [HumanMessage(content="(System: Error - Buyer configuration missing.)")]}

    # Get the LLM
    llm = get_llm()

    # Create the prompt dynamically using the loaded buyer prompt
    buyer_system_prompt = f"""
{current_buyer_prompt}

You are acting as the customer described above. Continue the conversation naturally based on the history provided below.
Focus on your persona's goals and interests. Ask questions or make statements consistent with your profile.
Your response should be just your conversational reply, without any preamble like "Buyer:".

Conversation History:
{{messages}}
"""
    buyer_prompt = ChatPromptTemplate.from_template(buyer_system_prompt)
    buyer_runnable = buyer_prompt | llm  # Buyer doesn't have tools

    # Check that messages exist and are not empty
    messages = state.get("messages", [])
    if not messages:
        print("Warning: No messages found in state for buyer. Adding a default message.")
        messages = [AIMessage(content="Hello, I'm a sales representative. How can I help you today?")]

    try:
        response = buyer_runnable.invoke({"messages": messages})
        buyer_response_content = response.content
        print(f"--- Buyer Response --- Type: {type(response)}")
        if buyer_response_content:
             print(f"Content: {buyer_response_content[:100]}...")

        # Return the message to be added to state
        return {"messages": [HumanMessage(content=buyer_response_content)]}

    except Exception as e:
        print(f"Error in buyer_node: {e}")
        
        # Attempt a simpler response as fallback
        try:
            print("Attempting fallback buyer response...")
            simple_llm = get_llm()  # Get a fresh LLM instance
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a customer. Ask a simple question about a product."),
                MessagesPlaceholder(variable_name="messages")
            ])
            simple_runnable = simple_prompt | simple_llm
            fallback_response = simple_runnable.invoke({"messages": messages[-3:] if len(messages) > 3 else messages})
            print(f"Generated fallback buyer response: {fallback_response.content[:100]}...")
            return {"messages": [HumanMessage(content=fallback_response.content)]}
        except Exception as fallback_error:
            print(f"Buyer fallback also failed: {fallback_error}")
            error_message = HumanMessage(content=f"(System: Encountered error in buyer response generation: {str(e)[:100]}...)")
            # Return error message
            return {"messages": [error_message]} 