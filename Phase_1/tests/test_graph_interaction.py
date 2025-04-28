import uuid
from pprint import pprint
import pytest # Uncommented pytest

from plan_sim.main import graph
from plan_sim.states import Plan, InputState, Assumptions
from plan_sim.config import Config
from plan_sim.env_tool import setup_environment
from langchain_core.runnables import RunnableConfig

# Ensure environment variables (like API keys) are loaded
# It's generally better practice to handle this with fixtures 
# or ensure the environment is set *before* running pytest,
# but for simplicity now, we'll leave it here.
try:
    setup_environment()
    print("Environment setup successful.")
except Exception as e:
    print(f"Error during environment setup: {e}")
    # pytest.fail(f"Environment setup failed: {e}") # Fail test if setup fails

# Rename the function to follow pytest convention
def test_run_leaky_faucet_scenario():
    """
    Tests the LangGraph graph with a predefined 'leaky faucet' scenario.
    """
    # Define inputs directly inside the test function
    initial_goal = "Successfully fix a leaky kitchen faucet."
    initial_assumptions = Assumptions(
        ground_truth=["Standard faucet parts are available.", "Basic plumbing tools are on hand.", "Water shut-off valve is functional."],
        vulnerabilities=["Pipes might be older and brittle.", "Unexpected corrosion found inside faucet."]
    )
    max_steps = 10 # Set the maximum number of steps
    topic = "Plumbing"

    print(f"\n--- Running Graph Interaction Test ---"
          f"\nGoal: {initial_goal}"
          f"\nTopic: {topic}"
          f"\nMax Steps: {max_steps}"
          f"\nInitial Assumptions:"
          f"\n Ground Truth: {initial_assumptions.ground_truth}"
          f"\nVulnerabilities: {initial_assumptions.vulnerabilities}"
          f"\n------------------------------------"
    )

    # Generate a unique run ID
    run_id = str(uuid.uuid4())

    # Prepare the initial state for the graph
    initial_input = InputState(goal=initial_goal, assumptions=initial_assumptions)
    initial_plan = Plan(
        plan_id=run_id,
        goal_state=initial_input, 
        input_state=[initial_input], 
        cumulative_assumptions=Assumptions(ground_truth=[], vulnerabilities=[]), # Initialize properly
        metric_count_1=100.0,
        metric_count_2=1.0    
    )

    # Prepare the configuration for the run
    run_config = RunnableConfig(
        recursion_limit=max_steps,
        configurable={
            "run_id": run_id,
            "topic": topic,
        }
    )

    try:
        # Stream the graph execution
        print("Starting graph stream...")
        final_state_update = None
        step_count = 0
        for step_output in graph.stream(initial_plan.model_dump(), config=run_config):
            step_count += 1
            node_name = list(step_output.keys())[0]
            node_output = step_output[node_name]
            print(f"\n<<< Step {step_count} Output from Node: {node_name} >>>")
            pprint(node_output, indent=2)
            final_state_update = step_output # Keep track of the last message

        print(f"\n--- Graph Stream Finished after {step_count} steps ---")

        if final_state_update:
             print("\nLast recorded state update:")
             pprint(final_state_update, indent=2)
        else:
             print("Graph did not produce any output steps (possibly finished immediately or hit limit without output).")
        
        # Basic assertions (enable and refine as needed)
        assert step_count > 0, "Graph did not execute any steps."
        assert final_state_update is not None, "Graph finished without any state update."

    except Exception as e:
        print(f"\n--- Error during graph execution ---")
        print(f"Error: {e}")
        pytest.fail(f"Graph execution failed: {e}") # Use pytest.fail

# Remove the __main__ block as pytest will discover the test function

# --- Add more test functions for different scenarios as needed ---
def test_run_frame_house_scenario():
    """
    Tests the LangGraph graph with a predefined 'frame house' scenario.
    """
    # Define complex inputs for building a frame house
    initial_goal = "Successfully construct the complete frame for a 2000 sq ft single-story residential house according to standard building codes."
    initial_assumptions = Assumptions(
        ground_truth=["Foundation is poured and cured.", "Building plans are approved.", "Lumber package has been delivered to site.", "Basic carpentry tools are available."],
        vulnerabilities=["Weather delays (rain, wind).", "Lumber quality issues (warping, knots).", "Potential for measurement errors during cutting/assembly.", "Crew availability fluctuations."]
    )
    max_steps = 30 # Reverted max steps back to 30 for complexity check
    topic = "Residential Construction Framing"
    initial_budget = 50000.0
    initial_time = 30.0 # Example: days

    print(f"\n--- Running Graph Interaction Test ---"
          f"\nGoal: {initial_goal}"
          f"\nTopic: {topic}"
          f"\nMax Steps: {max_steps}"
          f"\nInitial Budget: {initial_budget}"
          f"\nInitial Time: {initial_time}"
          f"\nInitial Assumptions:"
          f"\n Ground Truth: {initial_assumptions.ground_truth}"
          f"\nVulnerabilities: {initial_assumptions.vulnerabilities}"
          f"\n------------------------------------"
    )

    # Generate a unique run ID
    run_id = str(uuid.uuid4())

    # Prepare the initial state for the graph
    initial_input = InputState(goal=initial_goal, assumptions=initial_assumptions)
    initial_plan = Plan(
        plan_id=run_id,
        goal_state=initial_input, 
        input_state=[initial_input], 
        cumulative_assumptions=Assumptions(ground_truth=[], vulnerabilities=[]), # Initialize properly
        metric_count_1=initial_budget,
        metric_count_2=initial_time
    )

    # Prepare the configuration for the run
    run_config = RunnableConfig(
        recursion_limit=max_steps,
        configurable={
            "run_id": run_id,
            "topic": topic,
        }
    )

    try:
        # Stream the graph execution
        print("Starting graph stream...")
        final_state_update = None
        step_count = 0
        for step_output in graph.stream(initial_plan.model_dump(), config=run_config):
            step_count += 1
            node_name = list(step_output.keys())[0]
            node_output = step_output[node_name]
            print(f"\n<<< Step {step_count} Output from Node: {node_name} >>>")
            pprint(node_output, indent=2)
            final_state_update = step_output # Keep track of the last message

        print(f"\n--- Graph Stream Finished after {step_count} steps ---")

        if final_state_update:
             print("\nLast recorded state update:")
             pprint(final_state_update, indent=2)
        else:
             print("Graph did not produce any output steps (possibly finished immediately or hit limit without output).")

        # Basic assertions (enable and refine as needed)
        assert step_count > 0, "Graph did not execute any steps."
        assert final_state_update is not None, "Graph finished without any state update."

    except Exception as e:
        print(f"\n--- Error during graph execution ---")
        print(f"Error: {e}")
        pytest.fail(f"Graph execution failed: {e}") # Use pytest.fail

# Remove the __main__ block as pytest will discover the test function

# --- Add more test functions for different scenarios as needed ---
# def test_run_thermostat_scenario():
#     # ... similar setup for thermostat goal ...
#     pass 