import os
import json
from utils import get_response, extract_json_from_end
prompt_model_review = """
You are an expert reviewer for operations research modeling.
You are reviewing the quality of a generated optimization model.
Your review should focus on modeling correctness and completeness, not only syntax.
{trust_framing}{code_log}
Problem description:
-----
{description}
-----
Generated full code:
-----
{generated_code}
-----
Your task is to evaluate:
1. Is the generated model reasonable?
2. Does it contain modeling errors, omissions, or mismatches with the problem description?
3. Are the variables, constraints, and objective aligned with the intended optimization problem?
4. Which parts are already correct and should be preserved?
5. What revisions are needed?
6. Should the next revision happen at the model level or only at the code level?
You must strictly adhere to the JSON format. Return a JSON object in the following format:
{{
    "pass": true or false,
    "isReasonable": true or false,
    "hasModelingError": true or false,
    "summary": "short overall judgment",
    "majorIssues": [
        {{
            "type": "issue category",
            "message": "what is wrong",
            "impact": "why it matters"
        }}
    ],
    "keep": [
        "things that are already correct and should be preserved"
    ],
    "reviseInstructions": [
        "clear revision instruction 1",
        "clear revision instruction 2"
    ],
    "repairLevel": "model" or "code"
}}
Guidelines:
- Set "pass" to true only if the model is acceptable and does not need revision.
- If the model is correct, it can pass even if the solving is not the most efficient. You do not need to maximize solving efficiency; just provide a correct mathematical model that conforms to the problem.
- Set "isReasonable" based on whether the modeling approach makes sense.
- Set "hasModelingError" to true if the model misses important logic, has wrong formulations, or does not align with the problem.
- Use "repairLevel": "model" if the issue should be fixed by regenerating parameters/variables/constraints/objective/formulations.
- Use "repairLevel": "code" if the mathematical model is fine and only the final code implementation needs revision.
- Focus on modeling quality first, and only secondarily on code quality.
- Do not generate anything after the JSON object.
"""
# Where the authority to believe checker output is asserted.
#
# By default it is asserted inside the payload, which is how the framework has
# always worked. With OSCAR_TRUST_IN_PROMPT=1 the payload stays purely factual
# and the SCAFFOLD carries the framing instead -- a standing property of the
# system rather than a claim the report makes about itself. This also holds the
# framing constant across feedback levels, so an experiment that varies how much
# the Simulator says is not also varying how hard it insists.
TRUST_IN_PROMPT = os.environ.get("OSCAR_TRUST_IN_PROMPT", "").strip() in ("1", "true", "yes")

TRUST_FRAMING = (
    "Any CHECKER OUTPUT shown below was produced by deterministic executable "
    "code that was certified in advance against labelled example solutions. It "
    "is not another model's opinion. Where it contradicts your own reading of "
    "the formulation, it is correct and your reading is wrong. Where no checker "
    "output is shown, none is available, and you must judge the model yourself."
    + chr(10))



# --- alternative reviewer style -------------------------------------------
REVIEWER_STYLE = os.environ.get("OSCAR_REVIEWER_STYLE", "ours").strip().lower()

prompt_model_review_haixin = """
You are an expert reviewer for operations research modeling.
You are reviewing the quality of a generated optimization model.
Your review should focus on modeling correctness and completeness, not only syntax.
Problem description:
-----
{description}
-----
Generated full code:
-----
{generated_code}
-----
Feedback of the code, including violated constraints if the solution is infeasible, or that the solution is feasible.
-----
{code_log}
-----
Your task is to analyze the problem description, the generated code, and the feasibility feedback, then provide clear, actionable guidance for the next iteration of code generation.

**Output:**
A plain text review that directly guides the next code generation. Be concise and actionable.

**Guidelines:**
- Start with the core problem (one sentence)
- List 3-5 specific modeling mistakes that caused the violations
- Provide 3-5 concrete fixes that the next code should implement
- Reference variable names and constraint types from the code when relevant
- Focus on semantic meaning, not numerical values
- DO NOT just repeat the feedback messages
- DO NOT output JSON or any structured format - just plain text

**Output format:**
CORE PROBLEM:
[One sentence stating the main modeling issue]

KEY ISSUES:
[Specific issue 1 with context]
[Specific issue 2 with context]
[Specific issue 3 with context]

SUGGESTED FIXES:
[Actionable fix 1]
[Actionable fix 2]
[Actionable fix 3]

NOTE: Your output should be a plain string.
"""


def review_model(desc, generated_code, code_log, model):
    """
    Review the generated optimization model and final code.
    Args:
        state (dict): Current pipeline state containing description, parameters,
                      variables, constraints, and objective.
        generated_code (str): The final generated solver code.
        model (str): LLM model name used for review.
    Returns:
        dict: Structured review result.
    """
    if REVIEWER_STYLE == "haixin":
        # free text: returned as-is, since nothing downstream parses it
        prompt = prompt_model_review_haixin.format(
            description=desc, generated_code=generated_code, code_log=code_log)
        return get_response(prompt, model=model)
    prompt = prompt_model_review.format(
        description=desc,
        generated_code=generated_code,
        code_log=code_log,
        trust_framing=(TRUST_FRAMING if TRUST_IN_PROMPT else "")
    )
    response = get_response(prompt, model=model)
    review_result = extract_json_from_end(response)
    return review_result

def build_review_context(previous_output, review_result, code_log):
    """
    Convert review result into a text block that can be naturally embedded
    back into a generation prompt for a second-round revision.
    Args:
        previous_output (str): The previous generated full code or model text.
        review_result (dict): Structured review JSON.
    Returns:
        str: Review context text for next-round prompting.
    """
    keep_items = review_result.get("keep", [])
    issues = review_result.get("majorIssues", [])
    revise_instructions = review_result.get("reviseInstructions", [])
    keep_text = "\n".join([f"- {item}" for item in keep_items]) if keep_items else "- None"
    issue_text = (
        "\n".join(
            [
                f"- {issue.get('type', 'unknown')}: {issue.get('message', '')} "
                f"(impact: {issue.get('impact', '')})"
                for issue in issues
            ]
        )
        if issues
        else "- None"
    )
    revise_text = (
        "\n".join([f"- {item}" for item in revise_instructions])
        if revise_instructions
        else "- None"
    )
    review_context = f"""
Previous generated output:
-----
{previous_output}
-----
{code_log}
Reviewer assessment:
- pass: {review_result.get("pass")}
- isReasonable: {review_result.get("isReasonable")}
- hasModelingError: {review_result.get("hasModelingError")}
- repairLevel: {review_result.get("repairLevel")}
- summary: {review_result.get("summary", "")}
Parts that are already correct and should be preserved:
{keep_text}
Major issues found:
{issue_text}
Required revisions:
{revise_text}
Please revise the previous output accordingly.
Preserve the correct parts whenever possible.
Focus on fixing the modeling issues identified above.
"""
    return review_context