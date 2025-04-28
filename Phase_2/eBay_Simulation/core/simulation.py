"""
Core simulation logic using LangGraph.

This module defines the state graph, nodes, edges, and the main `run_simulation` 
function that orchestrates the interaction between buyer and seller agents, 
handles tool usage, analyzes conversations for sales, calculates product ranks, 
and logs results to the database.
"""
import os
import sys
import uuid
import re
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Import from refactored modules
from core.state import SimulationState
from core.agents import seller_agent_node, buyer_agent_node, seller_tools
from core.analysis import analyze_conversation_for_sale
from utils.persona import get_available_persona_ids, load_persona_prompt, create_persona_placeholder
from utils.db import get_db_connection, log_message_to_db, log_sale_to_db
# Import new product search functions
from core.product_search import embed_product_description, search_product_index

# --- Constants and Configuration ---

# Determine Project Root
PROJECT_ROOT = Path(__file__).parents[1]
# Update INDEX_DIR to point directly to the personas directory
INDEX_DIR = PROJECT_ROOT / "data" / "personas"

# Maximum conversation turns
MAX_CONVERSATION_TURNS = 10  # Max turns for each agent
MAX_MESSAGES = MAX_CONVERSATION_TURNS * 2 + 1  # Seller starts, buyer responds, seller responds

# Load environment variables
def load_environment():
    """Load environment variables from .env file."""
    # Check if .env exists in current directory, else try project root
    env_path = PROJECT_ROOT / ".env"
    if not os.path.exists(env_path):
        print(f"No .env file found at {env_path}")
        return False
    
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")
    
    # Ensure the key environment variables are properly loaded
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables.")
        return False
    
    print(f"Environment loaded successfully. API key (first 10 chars): {google_api_key[:10]}...")
    return True

# --- Graph Node Definitions ---

def initialize_simulation(state: SimulationState, config: RunnableConfig) -> dict:
    """
    First node of the graph. Loads the buyer persona prompt and initializes the state.
    Uses persona_id from config if provided, otherwise uses a default.
    Determines the product index path based on the persona_id.
    """
    print("--- Initializing Simulation ---")
    
    # Get persona_id from config if available
    configurable_config = config.get("configurable", {})
    persona_id = configurable_config.get("persona_id")
    
    if not persona_id:
        print("--- No persona_id provided in config. Using default. ---")
        available_personas = get_available_persona_ids()
        if not available_personas:
            raise ValueError("No persona files found and no persona_id provided in config.")
        persona_id = available_personas[0]  # Default to the first available persona
        print(f"--- Using default persona_id: {persona_id} ---")
    
    try:
        from core.agents import DEFAULT_SELLER_SYSTEM_PROMPT
        buyer_prompt = load_persona_prompt(persona_id)
        print(f"--- Loaded Buyer Prompt for {persona_id} ---")
        
        # Determine product index path based on persona ID
        # Assuming format like topic_X
        topic_match = re.match(r"topic_(\d+)", persona_id)
        product_index_path = None
        if topic_match:
            topic_number = topic_match.group(1)
            # Update index file name pattern for CSV files
            index_file_name = f"persona_topic_{topic_number}_top_purchases_nmf.csv"
            potential_path = INDEX_DIR / index_file_name
            if potential_path.exists(): # Check if the index file actually exists
                product_index_path = str(potential_path)
                print(f"--- Found Product Index: {product_index_path} ---")
            else:
                print(f"--- Warning: Index file not found at {potential_path} ---")
        else:
            print(f"--- Warning: Could not determine topic number from persona_id '{persona_id}'. Index path not set. ---")
            
        # Create initial welcome message to ensure messages list isn't empty
        # This helps prevent the "contents is not specified" error
        initial_message = HumanMessage(content="Hello, I'm interested in your products.")
        messages = state.get("messages", [])
        if not messages:
            messages = [initial_message]
        
        # Return updates to be merged into the state
        return {
            "persona_id": persona_id,  # Store the persona_id in state
            "current_buyer_prompt": buyer_prompt,
            "initial_seller_prompt": DEFAULT_SELLER_SYSTEM_PROMPT,
            "product_index_path": product_index_path, # Store the determined index path
            # Ensure messages list exists and has at least one message
            "messages": messages,
        }
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise RuntimeError(f"Failed to initialize simulation: {e}")

# Rename node for clarity
def calculate_product_rank(state: SimulationState) -> dict:
    """
    Node to calculate the average rank of the sold item within its persona index based on semantic similarity.
    Uses the extracted item name (title).
    Triggered after analyze_sale confirms a sale.
    """
    print("--- Calculating Product Rank in Index ---") # Updated print
    
    # Check if a sale was actually completed and we have the necessary info
    if not state.get("sale_completed"):
        print("--- Skipping rank calculation: No sale detected. ---")
        return {}
        
    item_name = state.get("sold_item_name")
    index_path = state.get("product_index_path")
    
    if not item_name:
        print("--- Skipping rank calculation: No item name found for sold item. ---")
        return {}
    if not index_path:
        print("--- Skipping rank calculation: No product index path found in state. ---")
        return {}
        
    # 1. Embed the item name
    embedding = embed_product_description(item_name)
    
    if not embedding:
        print("--- Failed to generate embedding. Rank calculation aborted. ---")
        # Update the correct state field
        return { "product_avg_rank": None } 
        
    # 2. Search the index (function now returns average rank)
    avg_rank = search_product_index(embedding, index_path)
    
    print(f"--- Calculated Average Rank: {avg_rank} ---")
    
    # Return the average rank to update the state
    # Update the correct state field
    return { "product_avg_rank": avg_rank }

# --- Conditional Edges ---

def check_seller_action(state: SimulationState) -> Literal["tools", "buyer_agent_node", "analyze_sale", END]:
    """
    Determines the next step after the seller node based on:
    - If the seller called a tool, route to the tool node
    - If there are sales indicators in the conversation, route to the analyze_sale node
    - If we've reached message limit, route to analyze_sale before ending
    - Otherwise, route to the buyer node for their response
    """
    print("--- Checking Seller Action ---")
    messages = state.get("messages", [])
    if not messages:
        print("Warning: No messages found. Ending conversation.")
        return END
    
    last_message = messages[-1]
    
    # 1. Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("--- Routing: Seller -> Tools ---")
        return "tools"
    
    # 2. Check for potential sale indicators in the recent messages
    if len(messages) >= 3:  # Need at least a few messages for context
        # Analyze the last few messages for sale indicators
        recent_messages = messages[-min(5, len(messages)):]
        conversation_text = " ".join([
            msg.content for msg in recent_messages 
            if hasattr(msg, "content") and msg.content
        ])
        
        # Define sale indicators - words/phrases that suggest a sale might be happening
        sale_indicators = [
            "buy", "purchase", "order", "sold", "deal", 
            "payment", "confirm", "agree", "accept", 
            "checkout", "shopping cart", "add to cart", "pay",
            "price is good", "sounds good", "I'll take it"
        ]
        
        # Check if any sale indicators are present
        if any(indicator.lower() in conversation_text.lower() for indicator in sale_indicators):
            print("--- Sale indicators detected, routing to analyze_sale ---")
            return "analyze_sale"
    
    # 3. Check message limit before proceeding to buyer
    current_message_count = len(messages)
    print(f"--- Seller responded directly. Message count: {current_message_count}/{MAX_MESSAGES} ---")
    if current_message_count >= MAX_MESSAGES:
        print(f"--- Routing: Max messages reached, checking for sale before ending ---")
        return "analyze_sale"  # Analysis final result before ending
    else:
        print("--- Routing: Seller -> Buyer ---")
        return "buyer_agent_node"

def route_after_analysis(state: SimulationState) -> Literal["calculate_product_rank", "buyer_agent_node", END]:
    """Determines the next step after the analyze_sale node."""
    print("--- Routing After Analysis ---")
    
    # Check if sale completed and index exists
    if state.get("sale_completed") and state.get("product_index_path"):
        print("--- Routing: analyze_sale -> calculate_product_rank ---")
        # Route to the renamed node
        return "calculate_product_rank"
    
    # If no sale or no index, check message limit
    current_message_count = len(state.get("messages", []))
    if current_message_count >= MAX_MESSAGES:
        print(f"--- Routing: analyze_sale -> END (Max messages: {current_message_count}) ---")
        return END
    else:
        print(f"--- Routing: analyze_sale -> buyer_agent_node (No sale/index or not max messages) ---")
        return "buyer_agent_node"
        
# Rename routing function for clarity
def route_after_rank_calc(state: SimulationState) -> Literal["buyer_agent_node", END]:
    """Determines the next step after calculate_product_rank node."""
    print("--- Routing After Rank Calculation ---") # Updated print
    current_message_count = len(state.get("messages", []))
    if current_message_count >= MAX_MESSAGES:
        print(f"--- Routing: calculate_product_rank -> END (Max messages: {current_message_count}) ---")
        return END
    else:
        print(f"--- Routing: calculate_product_rank -> buyer_agent_node ---")
        return "buyer_agent_node"

# --- Graph Building ---

def build_simulation_graph():
    """Builds and returns the LangGraph for the sales simulation."""
    workflow = StateGraph(SimulationState)

    # Add nodes
    workflow.add_node("initialize", initialize_simulation)
    workflow.add_node("seller_agent_node", seller_agent_node)
    workflow.add_node("buyer_agent_node", buyer_agent_node)
    # Add the ToolNode using the seller_tools list
    tool_node = ToolNode(seller_tools)
    workflow.add_node("tools", tool_node)
    workflow.add_node("analyze_sale", analyze_conversation_for_sale)
    # Add the renamed node
    workflow.add_node("calculate_product_rank", calculate_product_rank) 

    # Define edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "seller_agent_node")

    # Conditional edge from seller
    workflow.add_conditional_edges(
        "seller_agent_node",
        check_seller_action,
        {
            "tools": "tools",
            "buyer_agent_node": "buyer_agent_node",
            "analyze_sale": "analyze_sale",
            END: END  # Allow direct ending if needed
        }
    )
    # Edge from tools back to seller (usually)
    # Tool output is added to messages, then seller runs again
    # Note: check_seller_action is crucial here to route correctly after tool use
    workflow.add_edge("tools", "seller_agent_node") 
    
    workflow.add_edge("buyer_agent_node", "seller_agent_node")

    # Conditional edge from analysis
    workflow.add_conditional_edges(
        "analyze_sale",
        route_after_analysis,
        {
            # Use the renamed node key
            "calculate_product_rank": "calculate_product_rank", 
            "buyer_agent_node": "buyer_agent_node",
            END: END
        }
    )
    
    # Conditional edge from rank calculation
    workflow.add_conditional_edges(
        "calculate_product_rank",
        # Use the renamed routing function
        route_after_rank_calc, 
        {
            "buyer_agent_node": "buyer_agent_node",
            END: END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    print("--- Simulation graph compiled successfully ---")
    return app

# --- Main Simulation Runner ---

def run_simulation(persona_id: Optional[str] = None, initial_messages: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
    """Runs a single sales simulation using the LangGraph.

    Args:
        persona_id: The specific persona ID to use for the buyer. If None,
                    defaults to the first available persona.
        initial_messages: An optional list of messages to start the conversation with.
                          If None, a default greeting is used.

    Returns:
        A dictionary representing the final state of the simulation, including
        messages, sale status, extracted details, calculated rank, etc.
        Returns {"error": "message"} if a critical error occurs (e.g., DB connection,
        LLM initialization, graph execution).
    """
    print(f"\n=== Starting Simulation Run: Persona='{persona_id or 'Default'}' ===")
    
    # Make sure the environment is loaded
    if not load_environment():
        print("--- Exiting due to environment loading failure. ---")
        return {"error": "Failed to load environment variables"}
    
    # Ensure we have persona data available
    create_persona_placeholder()
    
    # Validate persona availability
    available_personas = get_available_persona_ids()
    if not available_personas:
        print("Error: No persona files found in the specified directory.")
        return {"error": "No persona files found"}
    
    # Generate a unique ID for this simulation run
    simulation_id = str(uuid.uuid4())
    print(f"--- Simulation ID: {simulation_id} ---")
    
    # Get DB connection if possible
    db_conn = get_db_connection()
    if not db_conn:
        print("--- Warning: Database logging is not available for this simulation. ---")
    
    # Build the graph
    try:
        app = build_simulation_graph()
    except Exception as e:
        print(f"--- Error building simulation graph: {e} ---")
        if db_conn:
            db_conn.close()
        return {"error": f"Failed to build simulation graph: {e}"}
    
    # Prepare the config with persona_id if provided
    config = {"configurable": {}}
    if persona_id:
        config["configurable"]["persona_id"] = persona_id
    # Set recursion limit during invocation
    config["recursion_limit"] = 100
    # Ensure a thread_id is set for state retrieval
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = simulation_id # Use simulation_id as thread_id
    
    # Prepare initial state if initial_messages provided
    initial_state = {"messages": []} if initial_messages is None else {"messages": initial_messages}
    
    # Run the simulation
    try:
        latest_state = {} # Keep track of the most recent known complete state
        
        # Use for_each to process events and log messages during the run
        print("--- Starting Graph Stream ---")
        for event in app.stream(initial_state, config):
            # Try to find the most recent state dictionary within the event data
            # Events often have the node name as key, and the output/state as value
            current_state_in_event = None
            if len(event) == 1:
                event_data = event[next(iter(event))]
                # Check if the event data looks like our SimulationState structure
                if isinstance(event_data, dict) and "messages" in event_data:
                    current_state_in_event = event_data
                    latest_state = current_state_in_event # Update latest known state
            
            # Log messages to DB if connection available (using the state found in *this* event)
            if db_conn and current_state_in_event:
                message_index = len(current_state_in_event["messages"]) - 1
                if message_index >= 0:
                    msg = current_state_in_event["messages"][message_index]
                    persona_id_val = current_state_in_event.get("persona_id", "unknown")
                    log_message_to_db(
                        db_conn, 
                        simulation_id, 
                        persona_id_val, 
                        message_index, 
                        msg
                    )
        
        print("--- Finished Graph Stream ---")
        
        # Use the state captured during the stream as the final state
        final_state = latest_state 
        
        # Print captured final state for debugging
        if final_state:
            print(f"--- Final State Captured (Keys): {list(final_state.keys())} ---")
            print(f"--- Final Message Count: {len(final_state.get('messages', []))} ---")
        else:
            print("--- Warning: No state was captured during the stream. Final state might be inaccurate. ---")
            final_state = {} # Use empty dict to avoid downstream errors

        # Log sale information if a sale was completed (using the captured final_state)
        if db_conn and final_state.get("sale_completed", False):
            log_sale_to_db(db_conn, simulation_id, final_state.get("persona_id", "unknown"), final_state)
            
        # Commit all database transactions after simulation completes
        if db_conn:
            db_conn.commit()
            print("--- Database changes committed successfully. ---")
            db_conn.close()
            print("--- Database connection closed. ---")
        
        # Print summary based on final_state
        total_turns = len(final_state.get("messages", [])) if final_state else 0
        print(f"--- Simulation completed with {total_turns} final messages ---")
        
        # Include simulation metadata in the return
        if final_state:
            final_state["simulation_id"] = simulation_id
            final_state["run_timestamp"] = datetime.now().isoformat()
            return final_state # Return the captured state
        else:
            # If no state was captured, return an error object
            return {"error": "No final state captured from the simulation stream.", "simulation_id": simulation_id}
        
    except Exception as e:
        print(f"--- Error during simulation stream processing: {e} ---")
        # Log the exception details
        import traceback
        traceback.print_exc() 
        
        if db_conn:
            try:
                db_conn.rollback()
                print("--- Database changes rolled back due to error. ---")
                db_conn.close()
                print("--- Database connection closed after error. ---")
            except Exception as db_err:
                print(f"--- Error closing/rolling back DB connection: {db_err} ---")
        
        return {
            "error": f"Simulation failed during stream processing: {str(e)}",
            "simulation_id": simulation_id
        }

# --- Script Execution ---

if __name__ == "__main__":
    # Optional: Add argument parsing here to allow command-line options
    # For now, just use a default persona if available
    
    # Ensure we have persona data
    create_persona_placeholder()
    
    # Run with default parameters
    result = run_simulation()
    
    # Print results summary
    if "error" in result:
        print(f"Simulation error: {result['error']}")
        sys.exit(1)
    
    print("\n--- Simulation Results ---")
    messages = result.get("messages", [])
    print(f"Total messages: {len(messages)}")
    
    if result.get("sale_completed", False):
        print(f"SALE COMPLETED: {result.get('sale_details', 'No details available')}")
    else:
        print("No sale was completed in this simulation.")
    
    print("\n--- End of Simulation ---") 