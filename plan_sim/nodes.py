from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Dict
from plan_sim.config import Config
from plan_sim.states import InputState, NextStep, Assumptions, Plan
from plan_sim.prompts import GENERATE_IDEA, DECIDE_RESULT
from plan_sim.env_tool import setup_environment
from typing import Literal
from langchain_core.runnables import RunnableConfig
# Ensure API key is set
setup_environment()





def generate_next_idea(State: Plan, config: RunnableConfig) -> Plan:
    """
    Generate next idea using OpenAI API based on input state and update Plan.
    
    Args:
        State: Current Plan state
        config: Configuration object containing model settings
        
    Returns:
        Updated Plan with new NextStep added to steps list
    """
    configurable = Config.from_runnable_config(config)
    input_state = State.input_state[-1]
    model = ChatOpenAI(
        model=configurable.thinking_model,
        temperature=0
    )
    structured_llm = model.with_structured_output(NextStep)
    # Format the prompt with input state
    formatted_prompt = GENERATE_IDEA.format(
        input_state=input_state.model_dump()
    )
    
    # Get response from OpenAI
    next_step = structured_llm.invoke(formatted_prompt)
    
    # Update Plan state with new step
    
    return {"steps": [next_step]}

class Decider(BaseModel):
    decision: Literal["success", "failure"]

def decider(State: Plan, config: RunnableConfig) -> Plan:
    configurable = Config.from_runnable_config(config)
    model = ChatOpenAI(
        model=configurable.thinking_model,
        temperature=0
    )
    structured_llm = model.with_structured_output(Decider)
    formatted_prompt = DECIDE_RESULT.format(
        topic=configurable.topic,
        ground_truth=", ".join(State.input_state[-1].assumptions.ground_truth),
        vulnerabilities=", ".join(State.input_state[-1].assumptions.vulnerabilities),
        next_step=State.steps[-1].model_dump()
    )
    decision = structured_llm.invoke(formatted_prompt)
    return {"step_results": [decision.decision]}

def decide_mood(state: Plan) -> Literal["good_outcome", "bad_outcome"]:
    
    # Often, we will use state to decide on the next node to visit
    last_result = state.step_results[-1] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if last_result == "success":

        # 50% of the time, we return Node 2
        return "good_outcome"
    
    # 50% of the time, we return Node 3
    return "bad_outcome"
