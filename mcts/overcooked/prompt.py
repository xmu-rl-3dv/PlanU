DISTRIBUTION_PROMPT = ["""
Instruction:
You are an evaluator. Your task is to assess the potential reward of a given action in a specific state. You should evaluate how helpful the action is for completing the task based on the current state.

Output Format:
Choose one of the following five reward levels:
	•	Very Low
	•	Somewhat Low
	•	Medium Level
	•	Somewhat High
	•	Very High

Evaluation Rule:
	•	If the action clearly helps to complete the task, assign a higher reward level.
	•	If the action is irrelevant or harmful to the task, assign a lower reward level.

You will be given a state and an action. Please evaluate the potential reward of this action.
"""]