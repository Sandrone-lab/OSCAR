import os
import json
from utils import get_response, extract_json_from_end,get_response_revise
from review import review_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "run_dev")
DESC_PATH = os.path.join(BASE_DIR, "desc.txt")
DATA_PATH = os.environ.get("OSCAR_DATA_PATH") or os.path.join(BASE_DIR, "data.json")
# Known optimal TPD for the instance in DATA_PATH; None = never stop early on value.
_t = os.environ.get("OSCAR_TARGET_OBJ")
TARGET_OBJ = float(_t) if _t not in (None, "", "none", "None") else None
REVIEWER = os.environ.get("OSCAR_LARGE_MODEL", "qwen3.6-flash")
CODER = os.environ.get("OSCAR_SMALL_MODEL", "qwen3.5-flash")
FIRST_CODER = os.environ.get("OSCAR_SMALL_MODEL", "qwen3.5-flash")
SOLVER_TL=int(os.environ.get("OSCAR_SOLVER_TL", 240))     # solver time limit, seconds
PROC_TL=int(os.environ.get("OSCAR_PROC_TL", 600))         # whole-process limit, seconds

# Rule 10 of the Coder prompts. With a limit (the default, 240) the Coder is told to set
# it, word for word as before. OSCAR_SOLVER_TL=0 drops the instruction so the solver can
# run toward optimality; run_code.py then stops it just short of the process limit
# (OSCAR_SOLVER_TIME_CAP), so the program still lives to write its incumbent.
if SOLVER_TL > 0:
    _TIME_LIMIT_RULE = ('10. Set a solver time limit BEFORE solving, so the run always terminates:\n'
                        '        model.setParam(COPT.Param.TimeLimit, %d)\n'
                        '    If the solver stops at the limit with a feasible incumbent, write that\n'
                        '    incumbent to "solution.json" as usual.' % SOLVER_TL)
else:
    _TIME_LIMIT_RULE = ('10. Do not set a solver time limit; let the solver run until it proves optimality.\n'
                        '    If the solver nevertheless stops with a feasible incumbent, write that\n'
                        '    incumbent to "solution.json" as usual.')

def safe_call_review(desc, generated_code, code_log, model, max_retries=3):
    """try to use review_model"""
    for attempt in range(max_retries):
        try:
            result = review_model(desc, generated_code, code_log, model)
            return result
        except Exception as e:
            print(f"Review failed (Trial {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "analysis": "Review failed, please check model manually",
                    "suggestions": ["Check constraints", "Check variables"],
                    "errors": str(e)
                }
    return None

def extract_code_block(text):
    """Extract code from LLM"""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + len("```")
        end = text.find("```", start)
        return text[start:end].strip()
    return text.strip()

def format_hard_feedback(result):
    """
    Convert deterministic feasibility-check results into text feedback.
    This feedback is ground truth and should dominate LLM review opinions.
    """
    if result['feasible']:
        # A feasible incumbent that will not improve is usually not a weak
        # objective -- it is a model that CANNOT EXPRESS the better solution.
        # colombi2017 sits at 1897 against an optimum of 2844 because its MTZ
        # subtour elimination forbids reusing an arc, and the optimum reuses
        # twelve, one of them four times. No amount of re-weighting the
        # objective reaches 2844 while that constraint stands.
        #
        # "Redundant" was the wrong word to reach for: a redundant constraint
        # does not change the feasible set, so removing one cannot improve
        # anything. What has to be found is a constraint that is too TIGHT.
        #
        # This can be said safely because the Simulator checks every candidate
        # against the description: a constraint loosened past what the problem
        # allows comes back rejected. Questioning a constraint is cheap here;
        # only ignoring the description is expensive.
        return """
The solution is feasible -- every rule of the problem holds -- but its objective is not the best achievable. Improve the objective without breaking any rule.

If previous attempts have already failed to beat this objective, stop re-weighting the objective and look instead for a constraint that FORBIDS the better solution. A model can be perfectly valid and still be unable to express the optimum. Look in particular for:
  - an elimination or ordering constraint that rules out a structure the problem permits -- a subtour-elimination or sequencing scheme that forbids revisiting a node, reusing an arc, or splitting a quantity, when the description never forbids it;
  - a linking or big-M constraint whose constant is too small, silently capping a quantity the description leaves free;
  - an equality imposed where the description states "at least" or "at most";
  - a variable domain or index range that excludes values the description allows -- an integer where the description permits a fraction, a bound of 1 where the description sets no limit;
  - a constraint applied to every case when the description applies it to only some.

Re-read the description for each constraint you have written and ask what it forbids, not what it enforces. Removing or loosening a constraint the problem does not actually impose is the usual route to a better objective; if you loosen one the problem DOES impose, the solution will be rejected as infeasible, so state the rule you are relying on.
               """

    lines = []
    lines.append("HARD_FEEDBACK:")
    lines.append("HARD_STATUS: MODEL_INFEASIBLE")
    lines.append("The solution produced by the previous generated model violates the true problem constraints.")
    lines.append("The following violations are detected by the deterministic feasibility checker.")
    lines.append("These violations are ground truth and must not be ignored or contradicted by the LLM reviewer.")
    lines.append("")
    lines.append("VIOLATED_TRUE_CONSTRAINTS:")

    violations = result.get("violations", [])
    if not violations:
        lines.append("1. The feasibility checker reports infeasibility, but no detailed violation message was returned.")
    else:
        for idx, v in enumerate(violations, start=1):
            if isinstance(v, dict):
                name = v.get("name") or v.get("constraint") or v.get("constraint_name") or f"Constraint {idx}"
                description = v.get("description") or v.get("message") or v.get("text") or ""
                detail = v.get("detail") or v.get("details") or ""

                lines.append(f"{idx}. True constraint violated: {name}")
                if description:
                    lines.append(f"   Description: {description}")
                if detail:
                    lines.append(f"   Detail: {detail}")
            else:
                lines.append(f"{idx}. True constraint violated: {v}")

    lines.append("")
    lines.append("MANDATORY_REVISION_INSTRUCTION:")
    lines.append("In the next generation, treat HARD_STATUS: MODEL_INFEASIBLE as a hard failure of the previous mathematical model.")
    lines.append("The next model must be changed so that the violated true constraints are explicitly respected.")
    lines.append("This is not a syntax-only issue. Do not merely patch the code unless the violation is caused by output-format or variable-extraction errors.")

    return "\n".join(lines)

def build_combined_review_feedback(hard_feedback, llm_feedback):
    """
    Combine deterministic feasibility feedback and LLM model-review feedback.
    The hard feedback has priority; the LLM feedback is only advisory.
    """
    return f"""
ITERATION_REVIEW_CONTEXT:

{hard_feedback}

LLM_REVIEW_OPINION:
The following is the LLM reviewer's diagnostic opinion.
It may help identify the likely modeling mistake, but it is not ground truth.
If it conflicts with HARD_FEEDBACK, the HARD_FEEDBACK must be followed.

{llm_feedback}

REVISION_PRIORITY:
1. First satisfy all violated true constraints listed in HARD_FEEDBACK.
2. Use LLM_REVIEW_OPINION only to infer which part of the generated model may have caused the violation.
3. If HARD_STATUS is MODEL_INFEASIBLE, the next attempt should modify the mathematical model, not only repair syntax.
4. Do not delete constraints merely to improve the objective value.
"""


def generate_structure(desc, model="qwen3.7-plus"):
    """Extract problem structure"""
    prompt = f"""
You are an expert in optimization modeling.

Analyze this optimization problem and extract the **structure** of all parameters needed.

PROBLEM DESCRIPTION:
{desc}

For each parameter, specify:
- name: CamelCase string
- type: "scalar", "vector", or "matrix"  
- shape: "[]" for scalar, "[N]" for vector, "[N, M]" for matrix
  (use parameter names like N, M, NumProducts, NumFactories for dimensions)
- description: brief natural language description

Output ONLY a JSON object in this format:
{{
    "ParameterName": {{
        "type": "scalar|vector|matrix",
        "shape": "[]" or "[N]" or "[N, M]",
        "description": "what this parameter means"
    }}
}}

Example:
{{
    "Capacity": {{
        "type": "scalar",
        "shape": "[]",
        "description": "maximum weight capacity of the knapsack"
    }},
    "Profit": {{
        "type": "vector",
        "shape": "[NumItems]",
        "description": "profit of each item"
    }}
}}

Do not generate anything after the JSON.
"""
    response = get_response(prompt, model)
    return extract_json_from_end(response)

def revise_code(desc, structure, data_keys, model, review_context="", previous_code=""):
    """Revise the code"""
    prompt = f"""
You are an expert in revising code with COPT (coptpy). Remember that the code is based on COPT solver so the generated code must strictly comply with the COPT Python API syntax specification.

PROBLEM DESCRIPTION:
{desc}

{_structure_block(structure)}

The actual data file "run_dev/data.json" contains the following parameters (keys):
{json.dumps(data_keys, indent=2)}

The actual data will be in a JSON file at "data.json" with exactly this structure.
All parameter values will be loaded from this file.

Here is the error message from the previous code:

{review_context}

Here is the previous code:

{previous_code}

You are revising the existing code based on the error message. This code may have other syntax errors. You need to identify as many as possible, rather than just looking at the error message snippet.

You may only modify the code to make it runnable; you should not modify the model. You can only revise the wrong part and NEVER modify the model. This is very important.

A complete WORKING Python code using COPT includes that:
1. Imports coptpy and sets up the model
2. Reads data from "data.json" using json.load()
3. Extracts all parameters with their proper shapes
4. Defines all decision variables (use CamelCase)
5. Sets the objective function (maximize or minimize)
6. Adds all constraints
7. Solves the model
8. Prints the optimal objective value
9. After solving, also save the complete solution to "solution.json"**
    with this EXACT structure:
    
    {{
        "variables":,
        "objective_value": <optimal objective value>
    }}
    ***It does not mean that the name is "variables"***. You should refer to the desc to set the name.
{_TIME_LIMIT_RULE}
11. If the model is infeasible, or the solver found no feasible solution at all,
    empty the "solution.json" file.
    with this reference code.
    if model.status == COPT.INFEASIBLE:    
        with open("solution.json", 'w') as f:
            pass


REQUIREMENTS:
- Use coptpy with: from coptpy import COPT
- Use COPT constants: COPT.MAXIMIZE, COPT.MINIMIZE, COPT.BINARY, COPT.INTEGER, COPT.CONTINUOUS
- Use cp.quicksum() for sums
- Use model.addVar() for scalars, model.addVars() for arrays
- Use model.addConstr() for constraints
- Use model.setObjective() with sense
- Use model.solve() to solve

CODE FORMAT:
Return ONLY the complete Python code in a markdown code block.

Example of expected code structure:
```python
import json
import coptpy as cp
from coptpy import COPT

env = cp.Envr()
model = env.createModel("OptimizationModel")

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract parameters
NumItems = data["NumItems"]
Capacity = data["Capacity"]
Profit = data["Profit"]
Weight = data["Weight"]

# Define variables
x = model.addVars(NumItems, vtype=COPT.BINARY, nameprefix="x")

# Objective
model.setObjective(cp.quicksum(Profit[i] * x[i] for i in range(NumItems)), sense=COPT.MAXIMIZE)

# Constraints
model.addConstr(cp.quicksum(Weight[i] * x[i] for i in range(NumItems)) <= Capacity)

# Solve
model.solve()
if model.status == COPT.OPTIMAL:
    print(f"Optimal value: {{model.objval}}")

Here are some common coding mistakes:
- Set one variable needs function model.addVar() and the type of variable is set by keyword argument "vtype".
- When you use function model.addVar(), the keyword argument is name. Do not use nameprefix.
- When you use function model.addVars(), the keyword argument is nameprefix. Do not use name or namePrefix.
- When you want to set the gap, use RelGap. Never use MIPGap, this is not avaliable in COPT.
- The attribute name is getAttr, not setAttr.
- Construct a environment is cp.Envr().
- When you want to set type, it is by COPT. For example, it is COPT.BINARY, not cp.BINARY.
- Solve statuses (model.status) include COPT.UNSTARTED(0), OPTIMAL(1), INFEASIBLE(2), UNBOUNDED(3), INF_OR_UNB(4), NUMERICAL(5), NODELIMIT(6), IMPRECISE(7), TIMEOUT(8), UNFINISHED(9), INTERRUPTED(10), ITERLIMIT(11) and LOCAL_OPTIMAL(20), and a code 12 that has no name and is returned when the solve stops at its memory limit. This is not a complete list you can branch over: do not write an if/elif chain over particular statuses whose final else exits or raises, because any status you did not name will crash the program. There is no COPT.NO_SOLUTION, no COPT.INFEASIBLE_OR_UNBOUNDED, no COPT.CUTOFF, no COPT.LOADED.
- To decide whether there is a solution to write, use model.hasmipsol for a MIP (model.haslpsol for an LP). Do NOT require COPT.OPTIMAL: a solve that hits the time limit reports TIMEOUT and may still hold a perfectly good incumbent, and writing no solution in that case throws it away.
- An environment is closed with env.close(). There is no closeEnv(). You do not need to close it at all; just let it go out of scope.
- addVars returns a tupledict indexed by the keys you built it with: model.addVars(A, B) is indexed x[a, b] (x[(a, b)] is the same thing and is equally fine). What does NOT work is a dict as a key -- "unhashable type: dict" means a dict reached an index position.
You SHOULD carefully check where the code had mistakes above! The usage of COPT is different from Gurobi.

CODE FORMAT:
Return ONLY the complete Python code in a markdown code block.
"""
    if model=="qwen3-coder-flash":
        response = get_response_revise(prompt)
    else:
        response = get_response(prompt,model)
    return extract_code_block(response)

OMIT_STRUCTURE = "__OMIT__"


def _structure_block(structure):
    """The DATA STRUCTURE section of a Coder prompt, or nothing at all.

    Passing OMIT_STRUCTURE drops the section entirely rather than sending an
    empty one: an empty block still tells the model a structure exists and was
    withheld, which is a different condition from never having had one.
    """
    if structure == OMIT_STRUCTURE:
        return ""
    header = "DATA STRUCTURE (parameters and their shapes):"
    return header + "\n" + json.dumps(structure, indent=2)


def generate_optimization_code(desc, structure, data_keys, model, attempt_history_text=""):
    """Generate the code"""
    prompt = f"""
You are an expert in optimization modeling and Python programming with COPT (coptpy).

PROBLEM DESCRIPTION:
{desc}

{_structure_block(structure)}

The actual data file "run_dev/data.json" contains the following parameters (keys):
{json.dumps(data_keys, indent=2)}

The actual data will be in a JSON file at "data.json" with exactly this structure.
All parameter values will be loaded from this file.

PAST ATTEMPTS, IN CHRONOLOGICAL ORDER:
{attempt_history_text}

When reading PAST ATTEMPTS:
- HARD_FEEDBACK is deterministic and authoritative.
- If HARD_STATUS: MODEL_INFEASIBLE appears, the previous generated model produced a solution violating true constraints.
- You must revise the mathematical model to satisfy the violated true constraints.
- LLM_REVIEW_OPINION is advisory and should be used only to locate the likely modeling error.
- Do not only make small syntax-level edits when the hard feedback indicates model infeasibility.

Your task:
Generate a NEW complete Python COPT model based on the history. You may need modify the model a lot.

Write COMPLETE, WORKING Python code using COPT that:
1. Imports coptpy and sets up the model
2. Reads data from "data.json" using json.load()
3. Extracts all parameters with their proper shapes
4. Defines all decision variables (use CamelCase)
5. Sets the objective function (maximize or minimize)
6. Adds all constraints
7. Solves the model
8. Prints the optimal objective value
9. After solving, also save the complete solution to "solution.json"**
    with this EXACT structure:
    
    {{
        "variables":,
        "objective_value": <optimal objective value>
    }}
    ***It does not mean that the name is "variables"***. You should refer to the desc to set the name.
{_TIME_LIMIT_RULE}
11. If the model is infeasible, or the solver found no feasible solution at all,
    empty the "solution.json" file.
    with this reference code.
    if model.status == COPT.INFEASIBLE:    
        with open("solution.json", 'w') as f:
            pass


REQUIREMENTS:
- Use coptpy with: from coptpy import COPT
- Use COPT constants: COPT.MAXIMIZE, COPT.MINIMIZE, COPT.BINARY, COPT.INTEGER, COPT.CONTINUOUS
- Use cp.quicksum() for sums
- Use model.addVar() for scalars, model.addVars() for arrays
- Use model.addConstr() for constraints
- Use model.setObjective() with sense
- Use model.solve() to solve

CODE FORMAT:
Return ONLY the complete Python code in a markdown code block.

Example of expected code structure:
```python
import json
import coptpy as cp
from coptpy import COPT

env = cp.Envr()
model = env.createModel("OptimizationModel")

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract parameters
NumItems = data["NumItems"]
Capacity = data["Capacity"]
Profit = data["Profit"]
Weight = data["Weight"]

# Define variables
x = model.addVars(NumItems, vtype=COPT.BINARY, nameprefix="x")

# Objective
model.setObjective(cp.quicksum(Profit[i] * x[i] for i in range(NumItems)), sense=COPT.MAXIMIZE)

# Constraints
model.addConstr(cp.quicksum(Weight[i] * x[i] for i in range(NumItems)) <= Capacity)

# Solve
model.solve()
if model.status == COPT.OPTIMAL:
    print(f"Optimal value: {{model.objval}}")

Here are some common coding mistakes:
- Set one variable needs function model.addVar() and the type of variable is set by keyword argument "vtype".
- When you use function model.addVar(), the keyword argument is name. Do not use nameprefix.
- When you use function model.addVars(), the keyword argument is nameprefix. Do not use name or namePrefix.
- When you want to set the gap, use RelGap. Never use MIPGap, this is not avaliable in COPT.
- The attribute name is getAttr, not setAttr.
- Construct a environment is cp.Envr().
- When you want to set type, it is by COPT. For example, it is COPT.BINARY, not cp.BINARY.
- Solve statuses (model.status) include COPT.UNSTARTED(0), OPTIMAL(1), INFEASIBLE(2), UNBOUNDED(3), INF_OR_UNB(4), NUMERICAL(5), NODELIMIT(6), IMPRECISE(7), TIMEOUT(8), UNFINISHED(9), INTERRUPTED(10), ITERLIMIT(11) and LOCAL_OPTIMAL(20), and a code 12 that has no name and is returned when the solve stops at its memory limit. This is not a complete list you can branch over: do not write an if/elif chain over particular statuses whose final else exits or raises, because any status you did not name will crash the program. There is no COPT.NO_SOLUTION, no COPT.INFEASIBLE_OR_UNBOUNDED, no COPT.CUTOFF, no COPT.LOADED.
- To decide whether there is a solution to write, use model.hasmipsol for a MIP (model.haslpsol for an LP). Do NOT require COPT.OPTIMAL: a solve that hits the time limit reports TIMEOUT and may still hold a perfectly good incumbent, and writing no solution in that case throws it away.
- An environment is closed with env.close(). There is no closeEnv(). You do not need to close it at all; just let it go out of scope.
- addVars returns a tupledict indexed by the keys you built it with: model.addVars(A, B) is indexed x[a, b] (x[(a, b)] is the same thing and is equally fine). What does NOT work is a dict as a key -- "unhashable type: dict" means a dict reached an index position.
You SHOULD carefully check where the code had mistakes above! The usage of COPT is different from Gurobi.
"""
    response = get_response(prompt, model)
    return extract_code_block(response)
