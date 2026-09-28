"""
Shared, crash-proof grading for both arms.

A generated program can exit 0 and still write a solution file that is not a
solution -- wrong schema, missing fields, wrong types. That is a real outcome of
the experiment, not an error in the harness, so it must be scored as infeasible
with a readable reason rather than raising.
"""

import problem

def validate_solution_schema(solution):
    """Return a list of plain-language schema problems; empty means well-formed.

    Field names come from problem.py, so this stays useful for another problem.
    """
    problems = []
    if not isinstance(solution, dict):
        return ["The solution file does not contain a JSON object."]

    field = problem.SOLUTION_LIST_FIELD
    if field is None:
        # heterogeneous solution formats: the trusted checker and the Simulator
        # judge structure; here we only require an object with the scalars
        for f in problem.SOLUTION_SCALAR_FIELDS:
            if f not in solution:
                problems.append('The solution file has no "%s" field.' % f)
        return problems
    items = solution.get(field)
    if items is None:
        problems.append(
            'The solution file has no "%s" field. It must be a JSON object with "%s" '
            "(a list of {%s} entries)%s. Found top-level keys: %s."
            % (field, field, ", ".join('"%s"' % f for f in problem.SOLUTION_ENTRY_FIELDS),
               "" if not problem.SOLUTION_SCALAR_FIELDS else
               ", plus " + ", ".join('"%s"' % f for f in problem.SOLUTION_SCALAR_FIELDS),
               sorted(solution.keys())))
        return problems
    if not isinstance(items, list):
        problems.append('"%s" must be a list, not %s.' % (field, type(items).__name__))
        return problems
    if not items:
        problems.append('"%s" is empty.' % field)

    for i, entry in enumerate(items[:5]):
        if not isinstance(entry, dict):
            problems.append("Entry %d of \"%s\" is not an object." % (i, field))
            continue
        missing = [f for f in problem.SOLUTION_ENTRY_FIELDS if f not in entry]
        if missing:
            problems.append("Entry %d of \"%s\" is missing %s." % (i, field, ", ".join(missing)))

    for f in problem.SOLUTION_SCALAR_FIELDS:
        if f not in solution:
            problems.append('The solution file has no "%s" field.' % f)
    return problems


def safe_check_feasibility(instance, solution):
    """check_feasibility that never raises.

    Returns the usual dict. A malformed solution comes back infeasible with
    constraint index 0 reserved for schema problems.
    """
    problems = validate_solution_schema(solution)
    if problems:
        return {"feasible": False, "status": "MALFORMED_SOLUTION",
                "violated_constraints": [0],
                "violations": problems, "violation_magnitudes": [],
                "schema_error": True}
    try:
        result = problem.check_feasibility(instance, solution)
        result.setdefault("schema_error", False)
        return result
    except Exception as e:
        return {"feasible": False, "violated_constraints": [0],
                "violations": ["The feasibility checker could not evaluate this solution: "
                               "%s: %s" % (type(e).__name__, e)],
                "violation_magnitudes": [], "schema_error": True}


def recompute_objective(instance, solution):
    """The true objective, recomputed from the decisions by problem.py."""
    return problem.recompute_objective(instance, solution)


# kept under its old name so existing call sites keep working
recompute_tpd = recompute_objective


def safe_simulate(simulator, instance, solution):
    """Run a generated simulator without letting it take the run down.

    Schema problems are reported before the simulator is called, so the Coder
    gets an actionable message rather than an opaque traceback.
    """
    # No schema pre-check here. The Simulator is given the syntax of the
    # operational decisions (Section 2.1) and is the component that decides
    # whether a candidate expresses a decision in that syntax. Intercepting it
    # here would take that judgement away from the oracle under study.
    # "__partial__" switches the simulator into fragment mode, which is used only
    # while the simulator is being certified. A candidate solution must never be
    # able to reach it, deliberately or by accident, so drop the key here.
    if isinstance(solution, dict) and "__partial__" in solution:
        solution = {k: v for k, v in solution.items() if k != "__partial__"}
    try:
        result = simulator(instance, solution)
        if not isinstance(result, dict) or "feasible" not in result:
            return {"feasible": False, "status": "SIMULATOR_ERROR",
                    "objective_value": None,
                    "violated_constraints": [0],
                    "violations": ["The simulator returned no usable verdict."],
                    "violation_magnitudes": [], "schema_error": False}
        result.setdefault("schema_error", False)
        return result
    except Exception as e:
        return {"feasible": False, "status": "SIMULATOR_ERROR",
                "objective_value": None,
                "violated_constraints": [0],
                "violations": ["The simulator could not evaluate this solution: "
                               "%s: %s" % (type(e).__name__, e)],
                "violation_magnitudes": [], "schema_error": False}
