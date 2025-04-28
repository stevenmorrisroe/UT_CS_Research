from plan_sim.env_tool import setup_environment
from plan_sim.nodes import generate_next_idea, decider, generate_good_outcome, generate_bad_outcome, route_result, goal_check, route_goal_check, abandon_check, route_abandon_check, summarize_assumptions
from plan_sim.states import Plan
from . import config
# Ensure API key is set
setup_environment()
from langgraph.graph import StateGraph, START, END

builder = StateGraph(Plan, config_schema=config.Config)


builder.add_node("generate_next_idea", generate_next_idea)
builder.add_node("decider", decider)
builder.add_node("generate_good_outcome", generate_good_outcome)
builder.add_node("generate_bad_outcome", generate_bad_outcome)
builder.add_node("goal_check", goal_check)
builder.add_node("abandon_check", abandon_check)
builder.add_node("summarize_assumptions", summarize_assumptions)



builder.add_edge(START, "generate_next_idea")
builder.add_edge("generate_next_idea", "decider")
builder.add_conditional_edges("decider", route_result)
builder.add_edge("generate_good_outcome", "goal_check")
builder.add_conditional_edges(
    "goal_check", 
    route_goal_check, 
    {
        "summarize_assumptions": "summarize_assumptions",
        END: END 
    }
)
builder.add_edge("generate_bad_outcome", "abandon_check")
builder.add_conditional_edges(
    "abandon_check", 
    route_abandon_check,
    {
        "summarize_assumptions": "summarize_assumptions",
        END: END
    }
)
builder.add_edge("summarize_assumptions", "generate_next_idea")


graph = builder.compile()
