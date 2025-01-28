# Goal state check prompt
GOAL_STATE_CHECK = """Given the current state and goal state, determine if the current state matches the goal state requirements.
Current state: {current_state}
Goal state: {goal_state}

Respond with only 'yes' or 'no'."""

# Idea/Assumption generation prompt
GENERATE_IDEA = """Based on the current state:
{input_state}

Generate a single concrete idea or assumption that could help reach the goal state. The idea should be:
1. Specific and actionable
2. Based on reasonable assumptions
3. Directly related to moving towards the goal state

Output format:
Idea: [your idea]
Assumptions: [list key assumptions]"""

# Force idea to work prompt
FORCE_IDEA = """Given this idea and its assumptions:
{idea}
{assumptions}

Describe specifically how this idea could be made to work, even if it seems challenging. Focus on:
1. Practical implementation steps
2. Required resources or conditions
3. Potential workarounds for obvious obstacles

Be concrete and specific in your response."""

# Predict outcome prompt
PREDICT_OUTCOME_WORKS = """Based on the following:
Idea: {idea}
Implementation: {implementation}
Current state: {current_state}

Predict the most likely outcome if this idea is implemented. Consider:
1. Direct effects
2. Potential side effects
3. Success probability
4. Possible failure modes

Provide a detailed prediction of the outcome."""


PREDICT_OUTCOME_FAILS = """Based on the following:
Idea: {idea}
Implementation: {implementation}
Current state: {current_state}

Predict the most likely outcome if this idea is implemented. Consider:
1. Direct effects
2. Potential side effects
3. Success probability
4. Possible failure modes

Provide a detailed prediction of the outcome."""
# Assumption validation prompt
VALIDATE_ASSUMPTIONS = """Compare the predicted outcome:
{predicted_outcome}

With the original assumptions:
{assumptions}

Determine if the outcome validates or invalidates these assumptions.
Respond with only 'yes' or 'no' followed by a brief explanation."""

# Lost cause assessment prompt
LOST_CAUSE_CHECK = """Given the following:
Original idea: {idea}
Attempted implementation: {implementation}
Actual outcome: {outcome}
Failed assumptions: {failed_assumptions}

Determine if this approach is a lost cause or if modifications could still make it viable.
Respond with only 'yes' (it's a lost cause) or 'no' (it can be modified) followed by a brief explanation."""

# Ground truth update prompt
UPDATE_GROUND_TRUTH = """Based on the results of our attempt:
Idea: {idea}
Outcome: {outcome}
Validated assumptions: {validated_assumptions}
Failed assumptions: {failed_assumptions}

Update our understanding of the problem space. Provide:
1. New confirmed facts or constraints
2. Disproven assumptions
3. Modified assumptions that better match reality
4. New potential approaches based on these learnings"""

# Vulnerability assessment prompt
VULNERABILITY_ASSESSMENT = """Review the following steps and assumptions:
Steps taken: {steps}
Current assumptions: {assumptions}

Identify potential vulnerabilities or weaknesses in our approach. Consider:
1. Hidden dependencies
2. Unstated assumptions
3. Potential points of failure
4. Environmental factors we might have overlooked

List specific vulnerabilities and explain their potential impact."""
