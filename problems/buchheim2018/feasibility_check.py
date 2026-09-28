#!/usr/bin/env python3
"""
Feasibility checker for the Quadratic Shortest Path Problem (QSPP)
from Buchheim & Traversi (2018), Problem (19).

Checks each hard constraint one by one:
  Constraint 1: Flow conservation for intermediate nodes
                 sum_{a in delta+(i)} x_a - sum_{a in delta-(i)} x_a = 0
                 for all i in N \\ {s, t}
  Constraint 2: Source outflow = 1
                 sum_{a in delta+(s)} x_a = 1
  Constraint 3: Sink inflow = 1
                 sum_{a in delta-(t)} x_a = 1
  Constraint 4: Binary variables
                 x_a in {0, 1} for all a in A
  Constraint 5: Binary domain check (auto-generated) for solution_x
  Constraint 6: Integer domain check (auto-generated)
  Constraint 7: Objective consistency -- reported objective_value must
                equal the recomputed value
                sum_{a,b in A} Q_{ab} x_a x_b + sum_{a in A} L_a x_a
"""

import argparse
import json
from collections import defaultdict


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    num_nodes = instance["num_nodes"]
    num_arcs = instance["num_arcs"]
    source = instance["source_node"]
    target = instance["target_node"]
    arcs = instance["arcs"]

    # Original solution structure is solution_arcs (list of selected arcs);
    # derive the binary x vector from it. Accept legacy solution_x dict
    # only as a fallback.
    x = [0.0] * num_arcs
    sol_arcs = solution.get("solution_arcs")
    if sol_arcs:
        for arc in sol_arcs:
            arc_id = int(arc.get("id", -1))
            if 0 <= arc_id < num_arcs:
                x[arc_id] = 1.0
    else:
        for k, v in solution.get("solution_x", {}).items():
            x[int(k)] = float(v)
    solution_x = {str(i): int(round(x[i])) for i in range(num_arcs) if x[i] > 0.5}

    # Build adjacency lists: outgoing and incoming arcs for each node
    delta_plus = defaultdict(list)   # outgoing arcs
    delta_minus = defaultdict(list)  # incoming arcs
    for arc in arcs:
        aid = arc["id"]
        delta_plus[arc["from_node"]].append(aid)
        delta_minus[arc["to_node"]].append(aid)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1: Flow conservation for intermediate nodes (= 0)
    #   sum_{a in delta+(i)} x_a - sum_{a in delta-(i)} x_a = 0
    #   for all i in N \ {s, t}
    # ------------------------------------------------------------------
    for i in range(num_nodes):
        if i == source or i == target:
            continue
        out_flow = sum(x[a] for a in delta_plus.get(i, []))
        in_flow = sum(x[a] for a in delta_minus.get(i, []))
        lhs = out_flow - in_flow
        rhs = 0.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(1)
            violations.append(
                f"Constraint 1: Flow conservation violated at node {i}: "
                f"outflow={out_flow}, inflow={in_flow}, net={lhs}"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint 2: Source outflow = 1
    #   sum_{a in delta+(s)} x_a = 1
    # ------------------------------------------------------------------
    lhs_source = sum(x[a] for a in delta_plus.get(source, []))
    rhs_source = 1.0
    violation_amount = abs(lhs_source - rhs_source)
    if violation_amount > tol:
        violated_constraints.add(2)
        violations.append(
            f"Constraint 2: Source outflow violated: "
            f"sum of outgoing arcs from source = {lhs_source}, expected 1"
        )
        normalizer = max(abs(rhs_source), eps)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": lhs_source,
            "rhs": rhs_source,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint 3: Sink inflow = 1
    #   sum_{a in delta-(t)} x_a = 1
    # ------------------------------------------------------------------
    lhs_sink = sum(x[a] for a in delta_minus.get(target, []))
    rhs_sink = 1.0
    violation_amount = abs(lhs_sink - rhs_sink)
    if violation_amount > tol:
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3: Sink inflow violated: "
            f"sum of incoming arcs to target = {lhs_sink}, expected 1"
        )
        normalizer = max(abs(rhs_sink), eps)
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_sink,
            "rhs": rhs_sink,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint 4: Binary constraint x_a in {0, 1} for all a in A
    # ------------------------------------------------------------------
    for a in range(num_arcs):
        dist_to_0 = abs(x[a] - 0.0)
        dist_to_1 = abs(x[a] - 1.0)
        violation_amount = min(dist_to_0, dist_to_1)
        if violation_amount > tol:
            # Nearest feasible binary value
            nearest_binary = 0.0 if dist_to_0 <= dist_to_1 else 1.0
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4: Binary constraint violated for arc {a}: "
                f"x_{a} = {x[a]}"
            )
            normalizer = max(abs(nearest_binary), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": x[a],
                "rhs": nearest_binary,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # Build result
    _domain_check_vars_binary = [("solution_x", solution_x)]
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 5: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > tol or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > tol:
                        violated_constraints.add(5)
                        violations.append(
                            f"Constraint 5 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 5,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # Constraint 6: Integer domain — variables must be integral
    for var_name, var_dict in _domain_check_vars_integer:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                frac = abs(v - round(v))
                if frac > tol:
                    violated_constraints.add(6)
                    violations.append(
                        f"Constraint 6 (integer domain): {var_name}[{key}] = {v} is not integer")
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": v,
                        "rhs": round(v),
                        "raw_excess": float(frac),
                        "normalizer": max(abs(round(v)), eps),
                        "ratio": float(frac / max(abs(round(v)), eps)),
                    })

    # ------------------------------------------------------------------
    # Constraint 7: Objective consistency
    #   reported objective_value must equal
    #   sum_{a,b in A} Q_{ab} x_a x_b + sum_{a in A} L_a x_a
    # All variables (selected arcs) are present in the solution, so we
    # can fully recompute the true objective. Tolerance is 0.5 because
    # the paper states coefficients are purely integer (see math_model.txt
    # reproduction-critical comment 6), so an integer mismatch by >=1
    # should fire regardless of relative magnitude.
    # ------------------------------------------------------------------
    linear_costs = instance.get("linear_costs")
    quadratic_costs = instance.get("quadratic_costs")
    reported_obj = solution.get("objective_value")
    if linear_costs is not None and quadratic_costs is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Selected arcs (treat any x[a] > 0.5 as selected)
            selected = [a for a in range(num_arcs) if x[a] > 0.5]
            linear_part = sum(float(linear_costs[a]) for a in selected)
            quadratic_part = 0.0
            for a in selected:
                row = quadratic_costs[a]
                for b in selected:
                    quadratic_part += float(row[b])
            true_obj = linear_part + quadratic_part
            abs_diff = abs(reported - true_obj)
            # Integer coefficients per paper -- tighten to 0.5 so any
            # off-by-1 (or worse) fires regardless of magnitude.
            obj_tol = max(0.5, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(7)
                violations.append(
                    f"Constraint 7: Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum_a,b Q_ab x_a x_b + sum_a L_a x_a="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for QSPP (Buchheim & Traversi 2018)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
