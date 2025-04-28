"""
Database utility functions for the sales simulation.

Handles connecting to the PostgreSQL database using environment variables
and provides functions to log simulation messages and completed sales records.
"""
import os
import psycopg2
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import Optional

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database.

    Reads connection parameters (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT) 
    from environment variables.

    Returns:
        A psycopg2 connection object, or None if the connection fails.
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        print("--- Database connection established successfully. ---")
        return conn
    except psycopg2.OperationalError as e:
        print(f"--- FATAL: Could not connect to database: {e} ---")
        print("--- Please check your .env file and PostgreSQL server status. ---")
        return None
    except Exception as e:
        print(f"--- FATAL: An unexpected error occurred during DB connection: {e} ---")
        return None

def log_message_to_db(conn, simulation_id: str, persona_id: str, turn_number: int, msg: BaseMessage):
    """Logs a single message from the simulation to the 'simulation_logs' table.

    Determines the role (Seller, Buyer, Tool, System) based on the message type.
    Handles specific fields like tool name and tool call ID.
    Truncates long tool message content.

    Args:
        conn: The active psycopg2 database connection.
        simulation_id: A unique identifier for the current simulation run.
        persona_id: The ID of the buyer persona used in this simulation.
        turn_number: The sequential turn number of this message within the simulation.
        msg: The Langchain BaseMessage object to log.
    """
    if not conn:
        print("--- Warning: Database connection not available. Skipping log. ---")
        return

    role = "System"  # Default
    content = msg.content
    tool_name = None
    tool_call_id = None

    if isinstance(msg, HumanMessage):
        role = "Buyer"
    elif isinstance(msg, AIMessage):
        role = "Seller"
        # Check for tool calls within the AIMessage
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            role = "Seller (Tool Call)"  # Distinguish seller turns that *request* a tool
            # Log details of the first tool call (can enhance later if multiple calls needed)
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_call_id = tool_call.get("id")
    elif isinstance(msg, ToolMessage):
        role = "Tool"
        tool_name = msg.name
        tool_call_id = msg.tool_call_id
        # Truncate potentially long tool content for logging
        if isinstance(content, str) and len(content) > 1000:
             content = content[:1000] + "... (truncated)"

    sql = """
        INSERT INTO simulation_logs (simulation_id, persona_id, turn_number, role, content, tool_name, tool_call_id, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        simulation_id,
        persona_id,
        turn_number,
        role,
        str(content),  # Ensure content is string
        tool_name,
        tool_call_id,
        datetime.now()  # Use current time for logging timestamp
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        # conn.commit() is handled after the loop in main
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"--- Error logging message turn {turn_number} to DB: {error} ---")
        conn.rollback()  # Rollback on error for this specific message

def log_sale_to_db(conn, simulation_id: str, persona_id: str, state):
    """Logs the details of a completed sale to the 'sales_records' table.

    Extracts sale information (item ID, name, price, rank, confidence, etc.) 
    from the final simulation state dictionary.
    Only logs if `state['sale_completed']` is True.

    Args:
        conn: The active psycopg2 database connection.
        simulation_id: The unique identifier for the simulation run.
        persona_id: The ID of the buyer persona used.
        state: The final simulation state dictionary containing sale details.
    """
    if not conn:
        print("--- Warning: Database connection not available. Skipping sale log. ---")
        return
    
    if not state.get("sale_completed", False):
        print("--- No sale detected, nothing to log to sales_records. ---")
        return
    
    sql = """
        INSERT INTO sales_records 
        (simulation_id, persona_id, sold_item_id, sold_item_name, sold_item_price, 
         product_avg_rank, is_verified, needs_review, sale_confidence, sale_details, sale_timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    params = (
        simulation_id,
        persona_id,
        state.get("sold_item_id"),
        state.get("sold_item_name"),
        state.get("sold_item_price"),
        state.get("product_avg_rank"),
        False,  # is_verified: default to false until human verification
        state.get("needs_review", True),
        state.get("sale_confidence", 0.0),
        state.get("sale_details"),
        state.get("sale_timestamp", datetime.now())
    )
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        # Note: We don't commit here; the commit will be handled at the end of the simulation
        print(f"--- Sale record prepared for simulation {simulation_id} ---")
        if state.get("needs_review", True):
            print(f"--- Note: This sale is flagged for human review (confidence: {state.get('sale_confidence', 0.0)}) ---")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"--- Error preparing sale record for DB: {error} ---")
        # Don't rollback here, as that would also rollback other messages 