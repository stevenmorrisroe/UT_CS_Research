from pydantic import BaseModel
from typing import List

class Assumptions(BaseModel):
    ground_truth: List[str]
    vulnerabilities: List[str]

class InputState(BaseModel):
    goal: str
    events: List[str]
    assumptions: Assumptions

    def to_dict(self) -> dict:
        return {
            "input_state": {
                "goal": self.goal,
                "events": self.events,
                "assumptions": self.assumptions.model_dump()
            }
        }

class NextStep(BaseModel):
    idea: str
    assumptions: List[str]

    def to_dict(self) -> dict:
        return {
            "next_step": {
                "Idea": self.idea,
                "assumptions": self.assumptions
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NextStep':
        return cls(
            idea=data['Idea'],
            assumptions=data['assumptions']
        )
class Plan(BaseModel):
    plan_id: str
    input_state: List[InputState]
    next_steps: List[NextStep]

    