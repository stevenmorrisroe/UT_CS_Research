from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Annotated, Any
from plan_sim.config import Config
from operator import add

def safe_append(existing: list, new: Any) -> list:
    """
    Appends new item(s) to the existing list.
    
    If `new` is a list, it concatenates it to the existing list.
    If `new` is a single item, it wraps it in a list and appends.
    """
    if isinstance(new, list):
        return existing + new
    else:
        return existing + [new]

class Assumptions(BaseModel):
    ground_truth: Annotated[List[str], add]
    vulnerabilities: Annotated[List[str], add]

class Outcome(BaseModel):
    new_truths: List[str]
    new_vulnerabilities: List[str]
    cost_increment: float
    time_increment: float
class NextStep(BaseModel):
    idea: str
    assumptions: List[str]


class InputState(BaseModel):
    goal: str
    assumptions: Assumptions
          


class Plan(BaseModel):
    plan_id: str
    goal_state: InputState
    input_state: Annotated[List[InputState], safe_append] = Field(default_factory=list)
    steps: Annotated[List[NextStep], safe_append] = Field(default_factory=list)
    outcomes: Annotated[List[Outcome], safe_append] = Field(default_factory=list)
    step_results: Optional[List[str]] = None
    metric_count_1: Optional[float] = 0
    metric_count_2: Optional[float] = 0
    cumulative_assumptions: Optional[Assumptions] = []
    final_outcome: Optional[Outcome] = None
    




