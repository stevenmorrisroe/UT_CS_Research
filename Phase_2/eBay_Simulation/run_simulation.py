#!/usr/bin/env python3
"""
Sales Conversation Simulator
Main entry point for running sales simulations.

Usage:
    python run_simulation.py [--persona PERSONA_ID] [--debug]
"""

import argparse
import sys
from typing import Optional, List
from pathlib import Path
import json

# Ensure core directory is in the Python path
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.simulation import run_simulation
from utils.persona import get_available_persona_ids, create_persona_placeholder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a sales conversation simulation")
    parser.add_argument(
        "--persona", 
        type=str, 
        help="Persona ID to use for the simulation (e.g., 'topic_0')"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug output"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save the simulation result JSON"
    )
    return parser.parse_args()

def main():
    """Main entry point for the simulation."""
    # Parse command line arguments
    args = parse_args()

    # Ensure we have persona data available
    create_persona_placeholder()

    # Get available personas
    available_personas = get_available_persona_ids()
    if not available_personas:
        print("Error: No persona files found. Make sure persona data is available in the data/personas directory.")
        return 1

    # Display available personas if in debug mode
    if args.debug:
        print(f"Available personas: {', '.join(available_personas)}")

    # Use specified persona or default to the first available
    persona_id = args.persona
    if not persona_id:
        persona_id = available_personas[0]
        print(f"No persona specified. Using default: {persona_id}")
    elif persona_id not in available_personas:
        print(f"Warning: Specified persona '{persona_id}' not found in available personas.")
        print(f"Available options: {', '.join(available_personas)}")
        return 1

    # Run the simulation
    print(f"Starting simulation with persona: {persona_id}")
    result = run_simulation(persona_id=persona_id)

    # Check for errors
    if "error" in result:
        print(f"Simulation error: {result['error']}")
        return 1

    # Print summary
    print("\n=== Simulation Results ===")
    messages = result.get("messages", [])
    print(f"Total messages: {len(messages)}")
    
    if result.get("sale_completed", False):
        print(f"SALE COMPLETED: {result.get('sale_details', 'No details available')}")
    else:
        print("No sale was completed in this simulation.")

    # Save output if requested
    if args.output:
        # Convert non-serializable objects to strings
        result_serializable = {}
        for key, value in result.items():
            if key == "messages":
                # Convert message objects to a readable format
                result_serializable[key] = [
                    {
                        "role": msg.__class__.__name__,
                        "content": msg.content if hasattr(msg, "content") else str(msg)
                    }
                    for msg in value
                ]
            else:
                # Convert other complex objects to strings if needed
                try:
                    json.dumps({key: value})  # Test if serializable
                    result_serializable[key] = value
                except (TypeError, OverflowError):
                    result_serializable[key] = str(value)
        
        # Save to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result_serializable, f, indent=2)
        print(f"Results saved to {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main()) 