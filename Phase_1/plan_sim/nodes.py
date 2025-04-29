import logging
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from plan_sim.config import Config
from plan_sim.states import InputState, NextStep, Assumptions, Plan, Outcome
from plan_sim.prompts import (GENERATE_IDEA, DECIDE_RESULT, PREDICT_OUTCOME_WORKS,
                              GOAL_STATE_CHECK, PREDICT_OUTCOME_FAILS, ABANDON_STATE_CHECK, SUMMARY_PROMPT)
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from plan_sim.vector_store import is_idea_novel, store_idea
from plan_sim.llm_utils import invoke_structured_llm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- End Logging Setup ---

# Constants
MAX_IDEA_RETRIES = 5
RECENT_STEPS_CONTEXT = 3
RECENT_TRUTHS_CONTEXT = 2
MIN_ASSUMPTIONS_FOR_SUMMARY = 3 # Summarize if > 2 truths or > 2 vulnerabilities

def _get_current_assumptions(state: Plan) -> Assumptions:
    """Extracts the most current cumulative or initial assumptions from the state."""
    if state.cumulative_assumptions:
        return state.cumulative_assumptions
    elif state.input_state: # Check if input_state list is not empty
        return state.input_state[-1].assumptions
    else:
        return Assumptions(ground_truth=[], vulnerabilities=[])

def _get_recent_context(state: Plan) -> tuple[list[str], str, str]:
    """Extracts recent steps and truths for context in checks."""
    # Get last N steps
    recent_steps_data = []
    if state.steps:
        recent_steps_data = state.steps[-RECENT_STEPS_CONTEXT:] if len(state.steps) >= RECENT_STEPS_CONTEXT else state.steps
    steps_str = "\n".join(f"- {step.idea}" for step in recent_steps_data)

    # Get last N ground truths from cumulative assumptions
    recent_truths_data = []
    current_assumptions = _get_current_assumptions(state) # Use existing helper
    if current_assumptions.ground_truth:
        truths = current_assumptions.ground_truth
        recent_truths_data = truths[-RECENT_TRUTHS_CONTEXT:] if len(truths) >= RECENT_TRUTHS_CONTEXT else truths
    truths_str = ", ".join(recent_truths_data)
    
    return recent_steps_data, steps_str, truths_str

def generate_next_idea(State: Plan, config: RunnableConfig) -> Plan:
    """
    Generate next idea using OpenAI API and check for novelty using vector store.
    Retries up to MAX_IDEA_RETRIES times if similar ideas are generated.
    """
    configurable = Config.from_runnable_config(config)
    input_state = State.input_state[-1]
    
    # Initialize/get cumulative_assumptions - Use helper
    cumulative_assumptions = _get_current_assumptions(State) 
    
    ground_truth = cumulative_assumptions.ground_truth
    vulnerabilities = cumulative_assumptions.vulnerabilities
    
    # Compile list of previous ideas for prompt context
    previous_ideas = [step.idea for step in State.steps] if State.steps else []
    ideas_str = "\n".join(f"- {idea}" for idea in previous_ideas)
    
    # Format the prompt
    formatted_prompt = GENERATE_IDEA.format(
        topic=configurable.topic,
        ground_truth=", ".join(ground_truth),
        vulnerabilities=", ".join(vulnerabilities),
        ideas=ideas_str,
        goal=input_state.goal,
        metric_count_1=State.metric_count_1,
        metric_count_2=State.metric_count_2
    )
    
    retries = 0
    next_step = None 
    idea_is_novel = False

    while retries < MAX_IDEA_RETRIES:
        # Use llm_utils function
        next_step = invoke_structured_llm(
            model_name=configurable.thinking_model,
            temperature=0.8,
            prompt=formatted_prompt,
            output_model=NextStep,
            model_kwargs={"top_p": 0.1}
        )
        step_number = len(State.steps) + 1
        
        # Use vector_store function
        idea_is_novel = is_idea_novel(next_step.idea, step_number) # Use default threshold
        if idea_is_novel:
            logging.info(f"Generated novel idea for step {step_number}.")
            break 
        else:
            retries += 1
            logging.warning(f"Idea was too similar, retry {retries}/{MAX_IDEA_RETRIES}...")
            if retries >= MAX_IDEA_RETRIES:
                logging.warning(f"Max retries reached for idea generation. Proceeding with the last generated idea.")
                break # Proceed with the non-novel idea

    if next_step is None: 
        logging.error("Failed to generate any idea after max retries.")
        raise ValueError("Failed to generate any idea.") 

    # Store the final idea in Qdrant using vector_store function
    # Only store if novel, or if max retries reached (log warning in that case)
    step_number = len(State.steps) + 1 # Recalculate in case needed
    store_idea(next_step.idea, step_number)
    if not idea_is_novel: 
        logging.warning(f"Stored idea for step {step_number} might be similar to existing ones (max retries hit)." )
   
    # Return the updated state
    # Return cumulative_assumptions only if it was potentially updated (though this node doesn't seem to update it)
    # Let's stick to the original logic of returning it if it was potentially initialized.
    current_cumulative = _get_current_assumptions(State) # Get potentially initialized assumptions

    return {
        "steps": next_step, 
        "cumulative_assumptions": current_cumulative # Return the one used/initialized
    }

class Decider(BaseModel):
    reason: str
    decision: Literal["success", "failure"]

def decider(State: Plan, config: RunnableConfig) -> Plan:
    """Decides if the last step was a success or failure."""
    configurable = Config.from_runnable_config(config)
    # Use helper to get current assumptions
    current_assumptions = _get_current_assumptions(State)
    current_truths = current_assumptions.ground_truth
    
    formatted_prompt = DECIDE_RESULT.format(
        topic=configurable.topic,
        ground_truth=", ".join(current_truths), # Use current_truths
        metric_count_1=State.metric_count_1,
        metric_count_2=State.metric_count_2,
        next_step=State.steps[-1].model_dump()
    )
    # Use llm_utils function
    decision_output = invoke_structured_llm(
        model_name=configurable.thinking_model,
        temperature=0,
        prompt=formatted_prompt,
        output_model=Decider
    )
    return {"step_results": [decision_output.decision]}

def route_result(state: Plan) -> Literal["generate_good_outcome", "generate_bad_outcome"]:
    last_result = state.step_results[-1] 
    if last_result == "success":
        return "generate_good_outcome"
    return "generate_bad_outcome"

def generate_good_outcome(state: Plan, config: RunnableConfig) -> Plan:
    """Predicts the outcome assuming the last step succeeded."""
    configurable = Config.from_runnable_config(config)
    # Use helper to get current assumptions
    current_assumptions = _get_current_assumptions(state)
    current_truths = current_assumptions.ground_truth
    current_vulnerabilities = current_assumptions.vulnerabilities
    
    formatted_prompt = PREDICT_OUTCOME_WORKS.format(
        topic=configurable.topic,
        ground_truth=", ".join(current_truths), 
        vulnerabilities=", ".join(current_vulnerabilities), 
        idea=state.steps[-1].idea,
        assumptions=", ".join(state.steps[-1].assumptions),
        metric_count_1=state.metric_count_1,
        metric_count_2=state.metric_count_2
    )
    # Use llm_utils function
    outcome = invoke_structured_llm(
        model_name=configurable.thinking_model,
        temperature=0,
        prompt=formatted_prompt,
        output_model=Outcome
    )
    updated_truths = current_truths + outcome.new_truths
    updated_vulnerabilities = current_vulnerabilities + outcome.new_vulnerabilities

    return {"outcomes": state.outcomes + [outcome] if state.outcomes else [outcome],
            "cumulative_assumptions": Assumptions(
                ground_truth=updated_truths,
                vulnerabilities=updated_vulnerabilities
            ),
            "metric_count_1": state.metric_count_1 - outcome.cost_increment,
            "metric_count_2": state.metric_count_2 - outcome.time_increment
            }

def generate_bad_outcome(state: Plan, config: RunnableConfig) -> Plan:
    """Predicts the outcome assuming the last step failed."""
    configurable = Config.from_runnable_config(config)
    # Use helper to get current assumptions
    current_assumptions = _get_current_assumptions(state)
    current_truths = current_assumptions.ground_truth
    current_vulnerabilities = current_assumptions.vulnerabilities
    
    formatted_prompt = PREDICT_OUTCOME_FAILS.format(
        topic=configurable.topic,
        ground_truth=", ".join(current_truths), 
        vulnerabilities=", ".join(current_vulnerabilities), 
        idea=state.steps[-1].idea,
        assumptions=", ".join(state.steps[-1].assumptions),
        metric_count_1=state.metric_count_1,
        metric_count_2=state.metric_count_2
    )
    # Use llm_utils function
    outcome = invoke_structured_llm(
        model_name=configurable.thinking_model,
        temperature=0,
        prompt=formatted_prompt,
        output_model=Outcome
    )
    updated_truths = current_truths + outcome.new_truths
    updated_vulnerabilities = current_vulnerabilities + outcome.new_vulnerabilities

    return {"outcomes": state.outcomes + [outcome] if state.outcomes else [outcome],
            "cumulative_assumptions": Assumptions(
                 ground_truth=updated_truths,
                 vulnerabilities=updated_vulnerabilities
            ),
            "metric_count_1": state.metric_count_1 - outcome.cost_increment,
            "metric_count_2": state.metric_count_2 - outcome.time_increment
            }

class GoalChecker(BaseModel):
    achieved: Literal["yes", "no"]

def goal_check(State: Plan, config: RunnableConfig) -> Plan:
    """Checks if the goal has been achieved based on recent state."""
    configurable = Config.from_runnable_config(config)
    # Use helper to get recent context
    _, steps_str, truths_str = _get_recent_context(State)
    goal_assumptions_str = ", ".join(State.goal_state.assumptions.ground_truth)
    
    formatted_prompt = GOAL_STATE_CHECK.format(
        steps=steps_str,
        truths=truths_str,
        goal=State.input_state[-1].goal, 
        goal_assumptions=goal_assumptions_str
    )
    # Use llm_utils function
    goal_status = invoke_structured_llm(
        model_name=configurable.thinking_model,
        temperature=0,
        prompt=formatted_prompt,
        output_model=GoalChecker
    )
    return {"step_results": [goal_status.achieved]}

def route_goal_check(state: Plan) -> Literal["summarize_assumptions", END]:
    last_result = state.step_results[-1] 
    if last_result == "no":
        return "summarize_assumptions"
    return END

class AbandonChecker(BaseModel):
    abandon: Literal["abandon", "press on"]

def abandon_check(State: Plan, config: RunnableConfig) -> Plan:
    """Checks if the plan should be abandoned based on resources and progress."""
    configurable = Config.from_runnable_config(config)
    # Use helper to get recent context
    _, steps_str, truths_str = _get_recent_context(State)
    goal_assumptions_str = ", ".join(State.goal_state.assumptions.ground_truth)
    
    formatted_prompt = ABANDON_STATE_CHECK.format(
        steps=steps_str,
        truths=truths_str,
        goal=State.input_state[-1].goal, 
        goal_assumptions=goal_assumptions_str,
        metric_count_1=State.metric_count_1,
        metric_count_2=State.metric_count_2
    )
    # Use llm_utils function
    abandon_status = invoke_structured_llm(
        model_name=configurable.thinking_model,
        temperature=0,
        prompt=formatted_prompt,
        output_model=AbandonChecker
    )
    return {"step_results": [abandon_status.abandon]}

def route_abandon_check(state: Plan) -> Literal["summarize_assumptions", END]:
    last_result = state.step_results[-1] 
    if last_result == "press on":
        return "summarize_assumptions"
    return END

def summarize_assumptions(state: Plan, config: RunnableConfig) -> Plan:
    """Summarizes cumulative assumptions if they are numerous enough."""
    configurable = Config.from_runnable_config(config)
    # Use helper to get current assumptions
    current_assumptions = _get_current_assumptions(state)
    ground_truth = current_assumptions.ground_truth
    vulnerabilities = current_assumptions.vulnerabilities

    if len(ground_truth) < MIN_ASSUMPTIONS_FOR_SUMMARY and len(vulnerabilities) < MIN_ASSUMPTIONS_FOR_SUMMARY:
        return {"cumulative_assumptions": current_assumptions}

    last_step_idea = state.steps[-1].idea if state.steps else "No steps taken yet."

    formatted_prompt = SUMMARY_PROMPT.format(
        topic=configurable.topic,
        ground_truth="\n".join(f"- {truth}" for truth in ground_truth),
        vulnerabilities="\n".join(f"- {vuln}" for vuln in vulnerabilities),
        metric_count_1=state.metric_count_1, 
        metric_count_2=state.metric_count_2,
        last_steps=last_step_idea 
    )
    # Use llm_utils function
    summarized_assumptions = invoke_structured_llm(
        model_name=configurable.thinking_model,
        temperature=0,
        prompt=formatted_prompt,
        output_model=Assumptions 
    )
    return {"cumulative_assumptions": summarized_assumptions}
    