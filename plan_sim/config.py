import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated

@dataclass(kw_only=True)
class Config:
    """The configurable fields for the application."""
    run_id: Optional[str] = None
    topic: str = "Hacking"
    subtopic: Optional[str] = None
    thinking_model: str = "gpt-4o-mini"
    sm_model: Optional[str] = None
    

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Config":
        """Create a Config instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# Create a default config instance
config = Config()
