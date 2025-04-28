DECIDE_RESULT = """You are a world-class expert in {topic}, renowned for your practical, real-world insights. In this simulation, you have the freedom to determine which elements of the context truly reflect reality and which do not. Carefully evaluate the following information as if you were making decisions on an actual project:
The remaining budget and time are {metric_count_1} dollars and {metric_count_2} weeks.
Real-World Evidence:
{ground_truth}

Proposed Action:
{next_step}

Given the concrete facts above and acknowledging that some factors remain uncertain or uncontrollable, decide whether this proposed action would realistically succeed or fail. Rely on your judgment to simulate real-world conditions, weighing both the tangible evidence and the unpredictable elements. Your response should be a single word—either 'success' or 'failure'—that best captures the likely outcome in a realistic scenario.

"""

SUMMARY_PROMPT = """\nYou are an expert synthesizer tasked with distilling the planning context for a project in {topic}. Your goal is to produce a concise, fact-driven summary that eliminates redundancy and primes the system for generating unique, non-repetitive ideas.\nThe remaining budget and time are {metric_count_1} dollars and {metric_count_2} weeks.\nThe context is provided in two parts:\n- Ground Truths: Verified facts that are directly relevant to the plan.\n- Vulnerabilities: Potential weaknesses that could hinder success.\n\nInstructions:\n1. Consolidate and reframe the provided ground truths and vulnerabilities into a streamlined summary.\n2. Remove any overlapping or redundant details to ensure every point is distinct.\n3. Emphasize unique insights that will encourage innovative and novel idea generation.\n4. Ensure that the final summary is clear, comprehensive, and focused on actionable information.\n5. Eliminate elements that no longer apply to the current state of the plan, considering the last step taken.\n\nGROUND TRUTHS:\n{ground_truth}\n\nVULNERABILITIES:\n{vulnerabilities}\n\nLast step: {last_steps}\n\nPlease provide your final synthesized summary below.\n"""


# Goal state check prompt
GOAL_STATE_CHECK = """You are evaluating the progress of a plan.

Primary Goal: {goal}

Original Goal Assumptions: {goal_assumptions}

Current Ground Truths: {truths}

Recent Steps Taken: {steps}

Based *primarily* on the Current Ground Truths, has the Primary Goal '{goal}' been substantially achieved?
Respond with only 'yes' or 'no'.
"""


ABANDON_STATE_CHECK = """You are assessing whether to continue a plan for the goal: {goal}
Original Goal Assumptions: {goal_assumptions}

Current State:
Remaining Budget: {metric_count_1} dollars
Remaining Time: {metric_count_2} weeks
Recent Steps: {steps}
Recently Confirmed Truths: {truths}

Critically evaluate the situation. Consider the remaining resources (budget/time) and the progress made (recent truths vs. goal). Is it realistically feasible and worthwhile to continue pursuing the goal?
Imagine what a pragmatic expert would advise.
Respond with 'abandon' if resources are critically low OR if progress seems stalled despite effort.
Otherwise, respond with 'press on'.
"""

# Idea/Assumption generation prompt
GENERATE_IDEA = """You are a domain expert in {topic} and a strategic planner tasked with achieving the following goal: 
**{goal}**

Your task is to propose the next concrete step that addresses a gap not yet covered by previous steps. This step must:
1. Create a measurable, causal impact toward reaching the goal.
2. Leverage known ground truths and mitigate identified vulnerabilities.
3. Be clearly distinct from all prior steps.

You have {metric_count_1} dollars to spend and {metric_count_2} weeks to complete the project.
Ensure your idea is realistic and achievable within the given constraints and directly move toward the goal

Context:
- **Previous Ideas:** {ideas}
- **Known Ground Truths:** {ground_truth}
- **Identified Vulnerabilities:** {vulnerabilities}

Instructions:
1. Review the context carefully and identify a specific gap or bottleneck that remains unaddressed.
2. Propose a single, concrete, and actionable idea that fills this gap.
3. Clearly explain how your idea causally advances the plan:
   - Describe its direct impact on moving closer to the goal.
   - Detail which ground truths it leverages and which vulnerabilities it mitigates.
4. Verify that your idea is not a rehash of any previous idea. If it is similar to a previous step, refine it until it is distinctly novel.
5. List the key assumptions that must hold true for your idea to succeed. These should be concrete and directly tied to the action.

Output Format (strictly adhere to this):
Idea: [Your actionable idea here]
Assumptions: [A concise list of key assumptions here]
"""



# Predict outcome prompt
PREDICT_OUTCOME_WORKS = """You have years of experience in {topic} and have a deep understanding of the problem space.\nYou will be given a step in a plan that worked in the context given.\nContext is formatted in this structure: Known Before Idea, Idea, Idea Dependent Assumptions\nEvolve this context with new elements derived from the implications of the idea working.\nDO NOT MAKE THINGS UP. THE NEW ELEMENTS ADDED MUST BE DERIVED FROM THE IMPLICATION OF THE IDEA WORKING USING PURE DEDUCTION AND LOGIC\nThe remaining budget and time are {metric_count_1} dollars and {metric_count_2} weeks.\nKnown Before Idea:\nFacts: \n{ground_truth}\n\nPotential Vulnerabilities:\n{vulnerabilities}\n\nIdea:\n{idea}\n\nIdea Dependent Assumptions:\n{assumptions}\n\nAfter the idea works, what do we know as new ground truths, what are new vulnerabilities?\nRESPOND WITH A MAXIMUM OF 5 GROUND TRUTHS AND 5 VULNERABILITIES. BE CONCISE AND TO THE POINT.\n\nEstimate the realistic cost and time increments resulting *directly* from executing this specific idea ({idea}).\n- Consider typical costs/time involved in {topic} for such an action.\n- Simple checks might have low increments, while actions involving purchases or significant labor should have higher increments.\n- Increments must be non-negative numbers.\nProvide these estimates clearly.\n"""


PREDICT_OUTCOME_FAILS = """You have years of experience in {topic} and have a deep understanding of the problem space.\nYou will be given a step in a plan that failed in the context given.\nContext is formatted in this structure: Known Before Idea, Idea, Idea Dependent Assumptions\nEvolve this context with new elements derived from the implications of the idea failing.\nDO NOT MAKE THINGS UP. THE NEW ELEMENTS ADDED MUST BE DERIVED FROM THE IMPLICATION OF THE IDEA FAILING USING PURE DEDUCTION AND LOGIC\nThe remaining budget and time are {metric_count_1} dollars and {metric_count_2} weeks.\nKnown Before Idea:\nFacts: \n{ground_truth}\n\nPotential Vulnerabilities:\n{vulnerabilities}\n\nIdea:\n{idea}\n\nIdea Dependent Assumptions:\n{assumptions}\n\nAfter the idea fails, what do we know as new ground truths, what are new vulnerabilities?\nRESPOND WITH A MAXIMUM OF 5 GROUND TRUTHS AND 5 VULNERABILITIES. BE CONCISE AND TO THE POINT.\n\nEstimate the realistic cost and time increments resulting *directly* from executing this specific idea ({idea}).\n- Consider typical costs/time involved in {topic} for such an action.\n- Simple checks might have low increments, while actions involving purchases or significant labor should have higher increments.\n- Increments must be non-negative numbers.\nProvide these estimates clearly.\n"""

