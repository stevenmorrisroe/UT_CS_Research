from pydantic import BaseModel
from typing import List, Literal, Optional
from plan_sim.config import Config, config

class Assumptions(BaseModel):
    ground_truth: List[str]
    vulnerabilities: List[str]

class Outcome(BaseModel):
    result: str
    success: bool
    new_truths: List[str]
    new_vulnerabilities: List[str]
    metric_increment: float
    
class NextStep(BaseModel):
    idea: str
    assumptions: List[str]


class InputState(BaseModel):
    goal: str
    assumptions: Assumptions
          
class Events(BaseModel):
    input_state: InputState
    step: 'NextStep'
    outcome: Outcome

class Plan(BaseModel):
    plan_id: str
    input_state: List[InputState]
    steps: Optional[List[NextStep]] = None
    outcomes: Optional[List[Outcome]] = None
    step_results: Optional[List[Literal["success", "failure"]]] = None
    metric_count: Optional[float] = None
    final_outcome: Optional[Outcome] = None
    

class FinalOutcome(BaseModel):
    original_goal: str
    original_input_state: InputState
    final_outcome: Literal["success", "failure"]
    log: List[Events]
