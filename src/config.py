from pydantic import BaseModel
from typing import Optional

class Config(BaseModel):
    run_id: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    thinking_model: Optional[str] = None
    sm_model: Optional[str] = None

    def __str__(self):
        return f"Config(run_id={self.run_id}, topic={self.topic}, subtopic={self.subtopic}, thinking_model={self.thinking_model}, sm_model={self.sm_model})"