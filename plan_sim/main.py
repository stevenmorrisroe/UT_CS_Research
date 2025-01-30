from plan_sim.env_tool import setup_environment
from plan_sim.nodes import generate_next_idea, decider
from plan_sim.states import Plan
import config
# Ensure API key is set
setup_environment()
from langgraph.graph import StateGraph, START, END

builder = StateGraph(Plan, config_schema=config.Config)



builder.add_node("generate_next_idea", generate_next_idea)
builder.add_node("decider", decider)

builder.add_edge(START, "generate_next_idea")
builder.add_edge("generate_next_idea", "decider")
builder.add_edge("decider", END)

graph = builder.compile()