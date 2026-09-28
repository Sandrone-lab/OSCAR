"""
Feasibility checker for the Graph Coloring problem.

Mathematical formulation from:
Mehrotra & Trick (1996), "A Column Generation Approach for Graph Coloring",
INFORMS Journal on Computing 8(4):344-354.

Constraints are numbered top-to-bottom from the formulation in math_model.txt:
  Constraint 1 (VC adjacency):  x_{ik} + x_{jk} <= 1  for all (i,j) in E, k=1,...,K
      => No two adjacent vertices share the same color.
  Constraint 2 (VC assignment):  sum_k x_{ik} = 1  for all i in V
      => Every vertex is assigned exactly one color.
  Constraint 3 (VC binary):  x_{ik} in {0,1}  for all i in V, k=1,...,K
      => Binary variable domain (inherently satisfied by integer solution format).
  Constraint 6 (obj consistency):  reported_obj == |{ vc[i] : i in V }|
      => Reported objective_value must equal the number of distinct color
         labels actually used in `vertex_colors`. Tier-C defense against
         LLM exploits that fabricate the objective_value.

Only the original vertex-coloring (VC) constraints 1-3 are checked. The IS
set-covering reformulation constraints (Constraint 4: sum_{s: i in s} x_s >= 1;
Constraint 5: x_s in {0,1}) require `color_classes` (pattern variables) that
a generated algorithm operating on the VC solution structure may not produce.
The LP relaxation (Section 3), MWIS subproblem (Section 4), and the set
covering example (Section 5) are also not constraints on the final coloring.
"""

import argparse
import json


def check_feasibility(instance, solution):
    """
    Check feasibility of a candidate graph coloring solution.

    Returns a dict with:
      - feasible (bool)
      - violated_constraints (list of int)
      - violations (list of str)
      - violation_magnitudes (list of dict)
    """
    tol = 1e-5
    eps = 1e-5

    nodes = instance["graph"]["nodes"]
    edges = [tuple(e) for e in instance["graph"]["edges"]]
    node_set = set(nodes)

    vertex_colors = solution.get("vertex_colors", {})

    # Convert vertex_colors keys to int (JSON keys are strings)
    vc = {}
    for k, v in vertex_colors.items():
        vc[int(k)] = int(v)

    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    violated_constraints_set = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (VC adjacency): x_{ik} + x_{jk} <= 1 for all (i,j) in E, k
    # Meaning: No two adjacent vertices may share the same color.
    # For each edge (i,j), if both i and j are colored the same, this is violated.
    # LHS = x_{ik} + x_{jk} (= 2 if same color k), RHS = 1
    # violation_amount = LHS - RHS (for <= constraint)
    # =========================================================================
    for u, v in edges:
        if u in vc and v in vc and vc[u] == vc[v]:
            lhs = 2.0
            rhs = 1.0
            violation_amount = lhs - rhs  # = 1.0
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints_set.add(1)
                violations.append(
                    f"Constraint 1 violated: adjacent vertices {u} and {v} "
                    f"share color {vc[u]}"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # =========================================================================
    # Constraint 2 (VC assignment): sum_k x_{ik} = 1 for all i in V
    # Meaning: Every vertex must be assigned exactly one color.
    # Checked directly against the `vertex_colors` dict (primary VC structure).
    # LHS = number of colors assigned to vertex i (0 or 1 from vertex_colors),
    # RHS = 1
    # =========================================================================
    for node in nodes:
        if node not in vc:
            lhs = 0.0
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints_set.add(2)
                violations.append(
                    f"Constraint 2 violated: vertex {node} is not assigned any color"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # =========================================================================
    # Constraint 3 (VC binary): x_{ik} in {0, 1} for all i in V, k = 1,...,K
    # Inherently satisfied by the integer solution format (vertex_colors maps
    # each vertex to one color index). No check needed.
    # =========================================================================

    # =========================================================================
    # Constraint 4 (IS-1 coverage) and Constraint 5 (IS-2 binary) SKIPPED:
    # These belong to the IS set-covering reformulation (Section 2 of the
    # paper, used for column generation). A generated algorithm operating
    # on the VC solution structure produces `vertex_colors` (not pattern
    # variables `x_s` / `color_classes`), so these reformulation-only
    # constraints are not checked here.
    # =========================================================================

    # =========================================================================
    # Constraint 6 (objective consistency):
    #   reported_obj == |{ vc[i] : i in V, i in vc }|
    # Tier-C defense: the eval pipeline trusts solution["objective_value"];
    # LLM-evolved candidates have learned to lie about it. For graph
    # coloring the objective equals the number of distinct color labels
    # actually used in `vertex_colors`, so we can fully recompute it.
    # Tolerance is 0.5 because the objective is an integer color count, so
    # any off-by-one or larger mismatch fires.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(len(set(vc.values())))
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5
            if abs_diff > obj_tol:
                lhs = reported
                rhs = true_obj
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violated_constraints_set.add(6)
                violations.append(
                    f"Constraint 6 violated: reported objective_value={reported} "
                    f"differs from recomputed |distinct(vertex_colors)|={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    feasible = len(violated_constraints_set) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Graph Coloring (Mehrotra & Trick, 1996)"
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

    if result["feasible"]:
        print(f"FEASIBLE: No constraint violations found.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint(s) violated.")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
