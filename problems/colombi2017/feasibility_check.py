#!/usr/bin/env python3
"""
Feasibility checker for the DPRPP-IC (Directed Profitable Rural Postman Problem
with Incompatibility Constraints) using Formulation (A) from Colombi et al. (2017).

Checks constraints (1)-(10) as listed in math_model.txt, plus constraint (11):
objective-value consistency between the reported objective_value and a
recomputation from x/y/u (Tier C defense against LLM score-gaming).
"""

import json
import argparse
import math
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'served_arcs', 'tour_arcs')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ()
_FRONTIEROR_DERIVED_FIELD_RULES = ()
_FRONTIEROR_CONDITIONAL_FIELD_RULES = ()


def _frontieror_validate_solution_contract(solution):
    """Return a structured rejection when the public solution contract is absent."""
    violations = []
    if not isinstance(solution, dict):
        violations.append("solution must be a JSON object")
    else:
        missing = [
            field
            for field in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS
            if field not in solution or solution[field] is None
        ]
        if missing:
            violations.append(
                "missing required solution field(s): " + ", ".join(missing)
            )

        for discriminator, expected_value, fields in _FRONTIEROR_CONDITIONAL_FIELD_RULES:
            if solution.get(discriminator) != expected_value:
                continue
            conditional_missing = [
                field
                for field in fields
                if field not in solution or solution[field] is None
            ]
            if conditional_missing:
                violations.append(
                    "missing conditionally required solution field(s): "
                    + ", ".join(conditional_missing)
                )

        empty = [
            field
            for field in _FRONTIEROR_NONEMPTY_SOLUTION_FIELDS
            if field in solution and solution[field] in (None, [], {})
        ]
        if empty:
            violations.append(
                "empty required decision witness field(s): " + ", ".join(empty)
            )

        for target, operation, source, source_key in _FRONTIEROR_DERIVED_FIELD_RULES:
            try:
                reported = float(solution[target])
                if operation == "length":
                    expected = float(len(solution[source]))
                elif operation == "mapping_value":
                    expected = float(solution[source][source_key])
                else:
                    raise ValueError("unknown derived-field operation")
            except (KeyError, TypeError, ValueError, OverflowError):
                violations.append(
                    f"cannot derive {target} from decision witness field {source}"
                )
                continue
            if (
                reported != reported
                or expected != expected
                or reported in (float("inf"), float("-inf"))
                or expected in (float("inf"), float("-inf"))
                or abs(reported - expected) > 1e-6
            ):
                violations.append(
                    f"{target}={reported} does not match the decision-derived "
                    f"value {expected}"
                )

        if "objective_value" in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS:
            objective = solution.get("objective_value")
            if (
                isinstance(objective, bool)
                or not isinstance(objective, (int, float))
                or objective != objective
                or objective in (float("inf"), float("-inf"))
            ):
                violations.append("objective_value must be a finite JSON number")

    if not violations:
        return None
    return {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": violations,
        "violation_magnitudes": [],
    }


def _frontieror_write_unexpected_rejection(error):
    """Turn malformed candidate/checker exceptions into a normal rejection."""
    import json as _frontieror_json
    import sys as _frontieror_sys

    result_path = None
    for index, argument in enumerate(_frontieror_sys.argv):
        if argument == "--result_path" and index + 1 < len(_frontieror_sys.argv):
            result_path = _frontieror_sys.argv[index + 1]
            break
        if argument.startswith("--result_path="):
            result_path = argument.split("=", 1)[1]
            break
    if not result_path:
        return False
    try:
        with open(result_path, "w") as result_handle:
            _frontieror_json.dump(
                {
                    "feasible": False,
                    "violated_constraints": ["malformed_solution"],
                    "violations": [
                        "checker rejected malformed candidate: "
                        + type(error).__name__
                        + ": "
                        + str(error)[:500]
                    ],
                    "violation_magnitudes": [],
                },
                result_handle,
                indent=2,
            )
    except Exception:
        return False
    return True



def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []

    # -------------------------------------------------------------------------
    # Parse instance
    # -------------------------------------------------------------------------
    num_nodes = instance["num_nodes"]
    depot = instance["depot"]

    arc_cost = {}
    for arc in instance["arcs"]:
        arc_cost[(arc[0], arc[1])] = arc[2]
    all_arcs = set(arc_cost.keys())

    arc_profit = {}
    for pa in instance["profitable_arcs"]:
        arc_profit[(pa[0], pa[1])] = pa[2]
    profitable_arcs = set(arc_profit.keys())

    vi_nodes = set(instance.get("VI_nodes", []))
    if not vi_nodes:
        vi_nodes = set(i for (i, _) in profitable_arcs)

    strong_incomp = [(e[0], e[1]) for e in instance["strong_incompatibilities"]]
    weak_incomp = []
    weak_penalty = {}

    val = defaultdict(float)
    for (i, j), p in arc_profit.items():
        c = arc_cost.get((i, j), 0)
        val[i] += (p - c)

    gamma = instance.get("generation_parameters", {}).get("gamma", 0.01)

    for edge in instance["weak_incompatibilities"]:
        i, j = edge[0], edge[1]
        if len(edge) >= 3:
            c_bar = edge[2]
        else:
            c_bar = math.ceil(gamma * (val[i] + val[j]))
        weak_incomp.append((i, j))
        weak_penalty[(i, j)] = c_bar

    v_bar = set()
    for (i, j) in strong_incomp:
        v_bar.add(i)
        v_bar.add(j)
    for (i, j) in weak_incomp:
        v_bar.add(i)
        v_bar.add(j)

    profitable_from = defaultdict(list)
    for (i, j) in profitable_arcs:
        profitable_from[i].append((i, j))

    nodes = set(range(num_nodes))

    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for (i, j) in all_arcs:
        outgoing[i].append((i, j))
        incoming[j].append((i, j))

    # -------------------------------------------------------------------------
    # Parse solution: reconstruct x, y, z, u variables
    # -------------------------------------------------------------------------
    # x[i,j]: number of times arc (i,j) is traversed
    x = defaultdict(int)
    for ta in solution.get("tour_arcs", []):
        key = (ta["from"], ta["to"])
        x[key] = ta["count"]

    # y[i,j]: 1 if profitable arc is served
    y = {}
    for (i, j) in profitable_arcs:
        y[(i, j)] = 0
    for sa in solution.get("served_arcs", []):
        key = (sa["from"], sa["to"])
        if key in profitable_arcs:
            y[key] = 1

    # z[i]: 1 if at least one profitable arc leaving node i (in V_bar) is served
    z = {}
    for i in v_bar:
        z[i] = 0
    for (i, j) in profitable_arcs:
        if y.get((i, j), 0) == 1 and i in v_bar:
            z[i] = 1

    # u[i,j]: 1 if weak incompatibility penalty between i,j is paid
    # u should be 1 when both z[i]=1 and z[j]=1 (otherwise the constraint is violated)
    u = {}
    for (i, j) in weak_incomp:
        # Infer u: if both nodes are active, the penalty must be paid
        if z.get(i, 0) == 1 and z.get(j, 0) == 1:
            u[(i, j)] = 1
        else:
            u[(i, j)] = 0

    # -------------------------------------------------------------------------
    # Helper to record a violation
    # -------------------------------------------------------------------------
    def record_violation(constraint_idx, message, lhs, rhs, violation_amount):
        violations.append((constraint_idx, message))
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio)
        })

    # =========================================================================
    # Constraint (1): x_ij >= y_ij for (i,j) in R
    # =========================================================================
    for (i, j) in profitable_arcs:
        lhs = x[(i, j)]
        rhs = y[(i, j)]
        # This is a >= constraint: violation_amount = max(rhs - lhs, 0)
        violation_amount = max(rhs - lhs, 0)
        if violation_amount > tol:
            record_violation(
                1,
                f"Constraint (1): Served arc ({i},{j}) has y=1 but x={lhs} (arc not traversed)",
                lhs, rhs, violation_amount
            )

    # =========================================================================
    # Constraint (2): flow conservation at each node j in V
    # sum_{(j,i) in delta+(j)} x_ji = sum_{(i,j) in delta-(j)} x_ij
    # =========================================================================
    for j in nodes:
        out_flow = sum(x[(jj, k)] for (jj, k) in outgoing[j])
        in_flow = sum(x[(k, jj)] for (k, jj) in incoming[j])
        lhs = out_flow
        rhs = in_flow
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(
                2,
                f"Constraint (2): Flow imbalance at node {j}: outflow={out_flow}, inflow={in_flow}",
                lhs, rhs, violation_amount
            )

    # =========================================================================
    # Constraint (3): connectivity
    # sum_{(i,j) in delta+(S)} x_ij >= y_ks for S subset V\{0}, (k,s) in R(S)
    # Check: if y_ks=1, the tour must connect S to the depot.
    # We check by finding connected components of the tour graph and verifying
    # that every served arc is in the component containing the depot.
    # =========================================================================
    # Build directed graph from tour arcs
    adj = defaultdict(set)
    active_nodes = set()
    for (i, j), count in x.items():
        if count > 0:
            adj[i].add(j)
            adj[j].add(i)
            active_nodes.add(i)
            active_nodes.add(j)

    # Find weakly connected components via BFS
    visited = set()
    depot_component = set()
    components = []
    for node in active_nodes:
        if node in visited:
            continue
        comp = set()
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    queue.append(nb)
        if comp:
            if depot in comp:
                depot_component = comp
            components.append(comp)

    # Also add depot to its own component if it has no arcs
    if depot not in active_nodes:
        depot_component = {depot}

    # Check each served profitable arc: both endpoints must be in the depot component
    for (k, s) in profitable_arcs:
        if y[(k, s)] != 1:
            continue
        # S = V \ {depot component} that contains k and s
        if k not in depot_component or s not in depot_component:
            # The served arc is disconnected from the depot
            # The cut value (arcs leaving the component containing k,s) is 0
            # LHS of constraint (3) = 0, RHS = y_ks = 1
            record_violation(
                3,
                f"Constraint (3): Served arc ({k},{s}) is disconnected from depot (not in depot's connected component)",
                0.0, 1.0, 1.0
            )

    # =========================================================================
    # Constraint (4): y_ij <= z_i for i in V_bar, (i,j) in R
    # =========================================================================
    for i in v_bar:
        for (ii, j) in profitable_from.get(i, []):
            lhs = y[(ii, j)]
            rhs = z.get(i, 0)
            # This is a <= constraint: violation_amount = max(lhs - rhs, 0)
            violation_amount = max(lhs - rhs, 0)
            if violation_amount > tol:
                record_violation(
                    4,
                    f"Constraint (4): y_{{{ii},{j}}}={lhs} > z_{i}={rhs}",
                    lhs, rhs, violation_amount
                )

    # =========================================================================
    # Constraint (5): z_i + z_j <= 1 for {i,j} in E_1 (strong incompatibility)
    # =========================================================================
    for (i, j) in strong_incomp:
        lhs = z.get(i, 0) + z.get(j, 0)
        rhs = 1
        violation_amount = max(lhs - rhs, 0)
        if violation_amount > tol:
            record_violation(
                5,
                f"Constraint (5): Strong incompatibility violated: z_{i}={z.get(i,0)} + z_{j}={z.get(j,0)} = {lhs} > 1",
                lhs, rhs, violation_amount
            )

    # =========================================================================
    # Constraint (6): z_i + z_j - u_ij <= 1 for {i,j} in E_2 (weak incompatibility)
    # =========================================================================
    for (i, j) in weak_incomp:
        lhs = z.get(i, 0) + z.get(j, 0) - u.get((i, j), 0)
        rhs = 1
        violation_amount = max(lhs - rhs, 0)
        if violation_amount > tol:
            record_violation(
                6,
                f"Constraint (6): Weak incompatibility violated: z_{i}+z_{j}-u_{{{i},{j}}} = {lhs} > 1",
                lhs, rhs, violation_amount
            )

    # =========================================================================
    # Constraint (7): x_ij >= 0 integer for (i,j) in A
    # =========================================================================
    for (i, j) in all_arcs:
        val_x = x[(i, j)]
        # Check non-negativity
        if val_x < -tol:
            violation_amount = abs(val_x)
            record_violation(
                7,
                f"Constraint (7): x_{{{i},{j}}}={val_x} is negative",
                val_x, 0, violation_amount
            )
        # Check integrality
        rounded = round(val_x)
        int_violation = abs(val_x - rounded)
        if int_violation > tol:
            record_violation(
                7,
                f"Constraint (7): x_{{{i},{j}}}={val_x} is not integer",
                val_x, rounded, int_violation
            )

    # Also check that tour arcs are valid arcs in the instance
    for (i, j), count in x.items():
        if count > 0 and (i, j) not in all_arcs:
            record_violation(
                7,
                f"Constraint (7): Tour arc ({i},{j}) does not exist in the instance arc set",
                count, 0, float(count)
            )

    # =========================================================================
    # Constraint (8): y_ij in {0,1} for (i,j) in R
    # =========================================================================
    for (i, j) in profitable_arcs:
        val_y = y[(i, j)]
        if val_y not in (0, 1):
            violation_amount = min(abs(val_y), abs(val_y - 1))
            record_violation(
                8,
                f"Constraint (8): y_{{{i},{j}}}={val_y} is not binary",
                val_y, round(val_y), violation_amount
            )

    # Also check that served arcs are valid profitable arcs
    for sa in solution.get("served_arcs", []):
        key = (sa["from"], sa["to"])
        if key not in profitable_arcs:
            record_violation(
                8,
                f"Constraint (8): Served arc ({key[0]},{key[1]}) is not a profitable arc in the instance",
                1, 0, 1.0
            )

    # =========================================================================
    # Constraint (9): z_i in {0,1} for i in V_bar
    # =========================================================================
    for i in v_bar:
        val_z = z.get(i, 0)
        if val_z not in (0, 1):
            violation_amount = min(abs(val_z), abs(val_z - 1))
            record_violation(
                9,
                f"Constraint (9): z_{i}={val_z} is not binary",
                val_z, round(val_z), violation_amount
            )

    # =========================================================================
    # Constraint (10): u_ij in {0,1} for {i,j} in E_2
    # =========================================================================
    for (i, j) in weak_incomp:
        val_u = u.get((i, j), 0)
        if val_u not in (0, 1):
            violation_amount = min(abs(val_u), abs(val_u - 1))
            record_violation(
                10,
                f"Constraint (10): u_{{{i},{j}}}={val_u} is not binary",
                val_u, round(val_u), violation_amount
            )

    # =========================================================================
    # Constraint (11): objective-value consistency (Tier C defense).
    # The reported objective_value must equal the recomputed
    #   sum_{(i,j) in R} p_ij * y_ij
    #   - sum_{(i,j) in A} c_ij * x_ij
    #   - sum_{{i,j} in E_2} c_bar_ij * u_ij
    # within a 0.1% relative tolerance (with a 1e-3 absolute floor).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            profit_term = sum(arc_profit[(i, j)] * y[(i, j)] for (i, j) in profitable_arcs)
            # Use arc_cost.get(...) so x entries on non-instance arcs (already
            # flagged by constraint 7) don't crash the recompute.
            cost_term = sum(arc_cost.get((i, j), 0) * count for (i, j), count in x.items())
            penalty_term = sum(weak_penalty[(i, j)] * u[(i, j)] for (i, j) in weak_incomp)
            true_obj = float(profit_term - cost_term - penalty_term)
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(
                    11,
                    f"Constraint (11): Objective consistency violated: "
                    f"reported objective_value={reported} differs from recomputed "
                    f"sum(p*y) - sum(c*x) - sum(cbar*u) = {true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    reported, true_obj, abs_diff,
                )
        elif reported is not None and not math.isfinite(reported):
            # Non-finite reported objectives (inf/nan) are definitionally inconsistent
            # with any feasible solution's finite objective.
            record_violation(
                11,
                f"Constraint (11): Objective consistency violated: "
                f"reported objective_value={reported} is not finite",
                reported, 0.0, float("inf"),
            )

    # -------------------------------------------------------------------------
    # Build output
    # -------------------------------------------------------------------------
    violated_indices = sorted(set(c for c, _ in violations))
    violation_messages = []
    for idx in violated_indices:
        msgs = [msg for c, msg in violations if c == idx]
        violation_messages.append("; ".join(msgs))

    feasible = len(violated_indices) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": violation_magnitudes
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for DPRPP-IC (Colombi et al. 2017, Formulation A)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print("Solution is INFEASIBLE.")
        print(f"Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
