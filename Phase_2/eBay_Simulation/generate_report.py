#!/usr/bin/env python3
"""
Orchestrates sales simulation runs, generates sales wisdom using Gemini,
calculates metrics, and produces a final report.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Ensure core and utils are in the Python path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Import necessary components from the project
from core.simulation import run_simulation
from core.agents import get_llm # To get the Gemini LLM instance
from utils.persona import get_available_persona_ids, create_persona_placeholder

# --- Configuration ---
# NUM_RUNS_PER_PERSONA = 1 # Number of simulations to run for each persona (set to 1 for initial testing)
# PERSONAS_TO_RUN = 1 # Number of personas to test (set to 1 for initial testing)
# OUTPUT_REPORT_FILE = "simulation_report.md"
# SALES_WISDOM_MODEL = "gemini-1.5-flash-latest" # Or adjust if needed

# --- Argument Parsing ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sales simulations, generate wisdom, and create a report."
    )
    parser.add_argument(
        "--runs-per-persona",
        type=int,
        default=1,
        help="Number of simulations to run for each selected persona.",
    )
    parser.add_argument(
        "--personas",
        nargs='*', # 0 or more arguments
        type=str,
        help="List of specific persona IDs to run. If omitted, runs all available personas.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="simulation_report.md",
        help="Path to save the generated Markdown report.",
    )
    parser.add_argument(
        "--wisdom-model",
        type=str,
        default="gemini-2.5-pro-exp-03-2025",
        help="Gemini model to use for generating sales wisdom.",
    )
    # Add a debug flag if needed later
    # parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

# --- Helper Functions ---

def format_messages_for_llm(messages: List[BaseMessage]) -> str:
    """Formats the Langchain message list into a plain text string for the LLM."""
    formatted_lines = []
    for msg in messages:
        role = "Unknown"
        content = ""
        if hasattr(msg, 'content'):
            content = msg.content
        
        if hasattr(msg, 'role'): # Standard LCEL format
            role = msg.role.capitalize()
        elif isinstance(msg, HumanMessage): # LangGraph state often uses these directly
             role = "Buyer"
        elif isinstance(msg, SystemMessage):
             role = "System"
        # Add other message types as needed (e.g., AIMessage for Seller)
        # For now, rely on class name if role isn't explicit
        else:
            role = msg.__class__.__name__.replace("Message", "")

        formatted_lines.append(f"{role}: {content}")
    return "\n".join(formatted_lines)

def get_sales_wisdom(llm, message_stream_text: str) -> str:
    """
    Uses the provided LLM instance to generate sales wisdom from a message stream.
    """
    print("--- Generating Sales Wisdom --- ")
    # Define the meta-prompt
    meta_prompt = f"""
Analyze the following sales conversation transcript. Based *only* on this interaction, extract concise, practical pieces of generalizable sales wisdom or insights for this consumer. Focus on strategies, techniques, communication styles, or customer behavior patterns observed that are niche to the conversation.

Avoid simply summarizing the conversation. Provide concrete takeaways. If no specific wisdom can be extracted, state that clearly.

Conversation Transcript:
---
{message_stream_text}
---

Extracted Sales Wisdom:
"""
    
    try:
        # Use the llm's invoke method directly for text generation
        response = llm.invoke(meta_prompt)
        
        if hasattr(response, 'content'):
            print("--- Wisdom generated successfully. ---")
            return response.content.strip()
        else:
            print("--- Warning: LLM response did not contain expected content. ---")
            return "(Error: Could not extract content from LLM response)"

    except Exception as e:
        print(f"--- Error generating sales wisdom: {e} ---")
        # Consider adding more robust error handling or retries if needed
        return f"(Error during wisdom generation: {e})"

def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Generates a Markdown report from the collected data."""
    print("--- Generating Markdown Report --- ")
    lines = ["# Sales Simulation Report", f"Report generated on: {datetime.now().isoformat()}", ""]

    # Overall Metrics
    lines.append("## Overall Metrics")
    lines.append(f"- Total Runs Conducted: {report_data.get('total_runs_overall', 0)}")
    lines.append(f"- Total Successful Sales: {report_data.get('total_sales_overall', 0)}")
    lines.append(f"- Overall Conversion Rate: {report_data.get('overall_conversion_rate', 0.0):.2%}")

    # Display Overall AOV
    overall_aov = report_data.get('overall_aov', 0.0)
    if report_data.get('total_sales_overall', 0) > 0 and overall_aov > 0.0:
         lines.append(f"- Overall Average Order Value (AOV): ${overall_aov:.2f}")
    else:
         lines.append("- Overall Average Order Value (AOV): N/A")

    # Display Overall Average Rank
    overall_avg_rank = report_data.get('overall_avg_rank', 0.0)
    if report_data.get('total_sales_overall', 0) > 0 and overall_avg_rank > 0.0:
         lines.append(f"- Overall Average Sold Item Rank Score: {overall_avg_rank:.4f}")
    else:
         lines.append("- Overall Average Sold Item Rank Score: N/A")
    
    lines.append("")

    # Per-Persona Results
    lines.append("## Per-Persona Results")
    for persona_id, data in report_data.get("personas", {}).items():
        lines.append(f"### Persona: {persona_id}")
        lines.append(f"- Runs Conducted: {data.get('total_runs', 0)}")
        lines.append(f"- Successful Sales: {data.get('sales_count', 0)}")
        
        # Calculate Conversion Rate
        conversion_rate = data.get('conversion_rate', 0.0)
        lines.append(f"- Conversion Rate: {conversion_rate:.2%}")

        # Handle AOV display
        aov = data.get('aov', 0.0)
        if data.get('sales_count', 0) > 0 and aov > 0.0:
            lines.append(f"- Average Order Value (AOV): ${aov:.2f}")
        else:
            lines.append("- Average Order Value (AOV): N/A")
            
        # Handle Avg Rank display
        avg_rank = data.get('avg_rank', 0.0)
        if data.get('sales_count', 0) > 0 and avg_rank > 0.0: # Check if avg_rank is meaningfully calculated
             lines.append(f"- Average Sold Item Rank Score: {avg_rank:.4f}")
        else:
            # Display N/A if no sales, or if rank wasn't calculated (indicated by 0.0 or missing)
             lines.append("- Average Sold Item Rank Score: N/A")
        
        lines.append("")
        lines.append("#### Sales Wisdom Extracted:")
        for i, wisdom in enumerate(data.get('sales_wisdom', [])):
            lines.append(f"**Run {i+1}:**")
            lines.append(f"> {wisdom}")
            lines.append("")
        if not data.get('sales_wisdom'):
            lines.append("(No wisdom extracted for this persona)")
        lines.append("")

    return "\n".join(lines)

# --- Main Execution ---

def main():
    """Main function to run simulations and generate the report."""
    print("--- Starting Report Generation Process ---")
    
    # Parse command line arguments
    args = parse_args()
    print(f"--- Configuration: Runs/Persona={args.runs_per_persona}, Report File='{args.output_file}', Wisdom Model='{args.wisdom_model}' ---")
    
    # Load environment variables (.env file)
    if not load_dotenv():
        print("Warning: Failed to load .env file. Proceeding without it, but API keys might be missing.")
        # Decide if this should be a critical failure or just a warning
        # For now, treating as warning, but check if get_llm() handles missing key
        # return 1 # Uncomment this line to make it a fatal error
        
    # Ensure placeholder persona exists if needed
    create_persona_placeholder()

    # Get available personas
    available_personas = get_available_persona_ids()
    if not available_personas:
        print("Error: No persona files found. Cannot run simulations.")
        return 1

    print(f"Found available personas: {', '.join(available_personas)}")

    # Select personas to run based on arguments
    if args.personas:
        # Validate provided personas against available ones
        personas_to_process = [p for p in args.personas if p in available_personas]
        invalid_personas = [p for p in args.personas if p not in available_personas]
        if invalid_personas:
            print(f"Warning: The following specified personas were not found and will be skipped: {', '.join(invalid_personas)}")
        if not personas_to_process:
            print("Error: None of the specified personas were found. Aborting.")
            return 1
    else:
        # Default to running all available personas
        personas_to_process = available_personas
        print("--- Running for all available personas. ---")

    print(f"Processing {len(personas_to_process)} persona(s) for {args.runs_per_persona} run(s) each: {', '.join(personas_to_process)}")

    # Initialize LLM for wisdom generation
    try:
        # Check if model needs adjustment (though get_llm might handle this)
        # Ideally, get_llm would accept a model name parameter
        wisdom_llm = get_llm() # Consider modifying get_llm to accept args.wisdom_model
        if hasattr(wisdom_llm, 'model') and wisdom_llm.model != args.wisdom_model:
             print(f"--- NOTE: Requesting wisdom model '{args.wisdom_model}', but get_llm() provided '{wisdom_llm.model}'. Using the provided one. Modify core.agents.get_llm if specific model selection is required. ---")
             # TODO: Enhance get_llm in core/agents.py to accept a model name parameter
    except Exception as e:
        print(f"CRITICAL: Failed to initialize LLM: {e}")
        return 1

    # --- Data Collection ---
    all_results = {}
    total_runs_overall = 0
    total_sales_overall = 0
    grand_total_value = 0.0 # Accumulator for overall AOV
    grand_total_rank = 0.0  # Accumulator for overall avg rank

    for persona_id in personas_to_process:
        print(f"\n=== Processing Persona: {persona_id} ===")
        persona_results = {
            "runs": [],
            "sales_wisdom": [],
            "total_runs": 0,
            "sales_count": 0,
            "total_value": 0.0,
            "total_rank": 0.0,
        }

        for i in range(args.runs_per_persona):
            run_number = i + 1
            print(f"--- Running Simulation {run_number}/{args.runs_per_persona} for {persona_id} ---")
            
            # Run the simulation
            # Use persona_id config for the specific persona
            sim_result = run_simulation(persona_id=persona_id) 
            total_runs_overall += 1
            persona_results["total_runs"] += 1

            # Check for simulation errors
            if "error" in sim_result:
                print(f"Error in simulation run {run_number}: {sim_result['error']}")
                # Store minimal error info or skip? For now, just log.
                wisdom = f"(Simulation Error: {sim_result['error']})"
            else:
                 # Store the full result (optional, could be large)
                persona_results["runs"].append(sim_result) 

                # Process successful run
                messages = sim_result.get("messages", [])
                message_text = format_messages_for_llm(messages)
                
                # Generate Sales Wisdom
                wisdom = get_sales_wisdom(wisdom_llm, message_text)
                
                # Track sales metrics
                if sim_result.get("sale_completed", False):
                    persona_results["sales_count"] += 1
                    total_sales_overall += 1
                    price = sim_result.get("sold_item_price")
                    rank = sim_result.get("product_avg_rank") 
                    if price is not None:
                        persona_results["total_value"] += price
                    if rank is not None:
                         persona_results["total_rank"] += rank
                         
            persona_results["sales_wisdom"].append(wisdom)

        # Calculate metrics for this persona
        if persona_results["sales_count"] > 0:
            persona_results["aov"] = persona_results["total_value"] / persona_results["sales_count"]
            persona_results["avg_rank"] = persona_results["total_rank"] / persona_results["sales_count"]
        else:
            persona_results["aov"] = 0.0
            persona_results["avg_rank"] = 0.0
            
        if persona_results["total_runs"] > 0:
             persona_results["conversion_rate"] = persona_results["sales_count"] / persona_results["total_runs"]
        else:
             persona_results["conversion_rate"] = 0.0

        all_results[persona_id] = persona_results

        # Accumulate totals for overall metrics
        grand_total_value += persona_results["total_value"]
        grand_total_rank += persona_results["total_rank"]

    # --- Calculate Overall Metrics ---
    overall_conversion_rate = (total_sales_overall / total_runs_overall) if total_runs_overall > 0 else 0.0
    overall_aov = (grand_total_value / total_sales_overall) if total_sales_overall > 0 else 0.0
    overall_avg_rank = (grand_total_rank / total_sales_overall) if total_sales_overall > 0 else 0.0

    # --- Report Generation ---
    final_report_data = {
        "total_runs_overall": total_runs_overall,
        "total_sales_overall": total_sales_overall,
        "overall_conversion_rate": overall_conversion_rate,
        "overall_aov": overall_aov,
        "overall_avg_rank": overall_avg_rank,
        "personas": all_results
    }

    # Write the report to the specified file
    report_markdown = generate_markdown_report(final_report_data)
    output_path = Path(args.output_file) # Use args.output_file
    try:
        with open(output_path, 'w') as f:
            f.write(report_markdown)
        print(f"--- Report successfully generated: {output_path} ---")
    except IOError as e:
        print(f"--- Error writing report file {output_path}: {e} ---")
        return 1

    print("--- Report Generation Process Finished ---")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 