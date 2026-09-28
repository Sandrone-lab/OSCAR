"""
Feasibility checker for the Bin Packing Problem with Time Lags (BPPTL).
Based on Model 2 from Letelier, Clautiaux, Sadykov (2022).

Constraints (numbered top to bottom in the formulation):
  1. (2b): Each item assigned to exactly one bin-period pair.
  2. (2c): Bin capacity per (bin, period) pair.
  3. (2d): Time lag constraints for arcs between real items.
  4. (2e): x_{i,b,p} in {0,1}.
  5. (2f): u_{b,p} in {0,1}.
  6. Preprocessing: each real item's period lies in [es(i), ls(i)].
  7. Solution-level domain/consistency of active_bin_periods sub-fields.
  8. Objective consistency: the reported objective_value must equal the
     recomputed objective (2a) = sum_{b,p} u_{b,p}, i.e. the number of
     distinct bin-period pairs (gurobi format) / non-empty bins (efficient
     format) that contain at least one item. Tier C defense against
     fabricated objective_value exploits.
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def detect_solution_format(solution):
    """Detect whether solution is from gurobi or efficient algorithm."""
    if "assignments" in solution:
        return "gurobi"
    elif "bins" in solution:
        return "efficient"
    else:
        raise ValueError("Unknown solution format")


def extract_assignments_gurobi(solution):
    """Extract item -> (bin, period) mapping from gurobi solution."""
    assignments = {}
    for a in solution["assignments"]:
        item_id = a["item_id"]
        assignments[item_id] = {"bin": a["bin"], "period": a["period"], "weight": a["weight"]}
    return assignments


def extract_assignments_efficient(solution, instance):
    """Extract item -> bin mapping from efficient solution (no period info)."""
    weights = {it["item_id"]: it["weight"] for it in instance["items"]}
    assignments = {}
    for bin_idx, b in enumerate(solution["bins"]):
        for item_id in b["items"]:
            assignments[item_id] = {"bin": bin_idx, "weight": weights[item_id]}
    return assignments


def add_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, violation_amount):
    """Record a violation."""
    normalizer = max(abs(rhs), EPS)
    ratio = violation_amount / normalizer
    violations_list.append((constraint_idx, message))
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def check_feasibility(instance, solution):
    violations_list = []  # list of (constraint_idx, message)
    magnitudes_list = []

    fmt = detect_solution_format(solution)
    has_periods = (fmt == "gurobi")

    # Get real items from instance
    real_items = [it for it in instance["items"] if it["type"] == "treatment"]
    real_item_ids = set(it["item_id"] for it in real_items)
    weights = {it["item_id"]: it["weight"] for it in instance["items"]}
    W = instance["parameters"]["W"]
    T = instance["parameters"]["T"]

    # Extract assignments
    if has_periods:
        assignments = extract_assignments_gurobi(solution)
    else:
        assignments = extract_assignments_efficient(solution, instance)

    # ------------------------------------------------------------------
    # Constraint 1 (2b): Each real item assigned to exactly one bin(-period)
    # sum_{b,p} x_{i,b,p} = 1 for all i in V
    # ------------------------------------------------------------------
    assigned_items = set(assignments.keys())

    for it in real_items:
        iid = it["item_id"]
        if iid not in assigned_items:
            # LHS = 0 (not assigned), RHS = 1
            lhs_val = 0.0
            rhs_val = 1.0
            viol = abs(lhs_val - rhs_val)
            if viol > TOL:
                add_violation(violations_list, magnitudes_list, 1,
                              f"Item {iid} is not assigned to any bin",
                              lhs_val, rhs_val, viol)

    # Check for items assigned multiple times (shouldn't happen with dict, but
    # check original data for duplicates)
    if has_periods:
        item_count = {}
        for a in solution["assignments"]:
            iid = a["item_id"]
            item_count[iid] = item_count.get(iid, 0) + 1
        for iid, count in item_count.items():
            if iid in real_item_ids and count != 1:
                lhs_val = float(count)
                rhs_val = 1.0
                viol = abs(lhs_val - rhs_val)
                if viol > TOL:
                    add_violation(violations_list, magnitudes_list, 1,
                                  f"Item {iid} assigned {count} times (should be exactly 1)",
                                  lhs_val, rhs_val, viol)
    else:
        item_count = {}
        for b in solution["bins"]:
            for iid in b["items"]:
                item_count[iid] = item_count.get(iid, 0) + 1
        for iid, count in item_count.items():
            if iid in real_item_ids and count != 1:
                lhs_val = float(count)
                rhs_val = 1.0
                viol = abs(lhs_val - rhs_val)
                if viol > TOL:
                    add_violation(violations_list, magnitudes_list, 1,
                                  f"Item {iid} assigned {count} times (should be exactly 1)",
                                  lhs_val, rhs_val, viol)

    # Check for non-real items assigned
    for iid in assigned_items:
        if iid not in real_item_ids:
            lhs_val = 1.0
            rhs_val = 0.0
            viol = abs(lhs_val - rhs_val)
            if viol > TOL:
                add_violation(violations_list, magnitudes_list, 1,
                              f"Non-real item {iid} (source/sink) is assigned to a bin",
                              lhs_val, rhs_val, viol)

    # ------------------------------------------------------------------
    # Constraint 2 (2c): Capacity per (bin, period) pair
    # sum_i w_i x_{i,b,p} <= W * u_{b,p}
    # Since u_{b,p}=1 for active bins: sum weights <= W
    # ------------------------------------------------------------------
    if has_periods:
        # Group items by (bin, period)
        bin_period_items = {}
        for iid, asg in assignments.items():
            key = (asg["bin"], asg["period"])
            if key not in bin_period_items:
                bin_period_items[key] = []
            bin_period_items[key].append(iid)

        for (b, p), items in bin_period_items.items():
            total_w = sum(weights.get(iid, 0) for iid in items)
            # u_{b,p} = 1 if items present, so RHS = W
            lhs_val = float(total_w)
            rhs_val = float(W)
            viol = lhs_val - rhs_val
            if viol > TOL:
                add_violation(violations_list, magnitudes_list, 2,
                              f"Capacity exceeded in bin {b}, period {p}: "
                              f"total weight {total_w} > capacity {W}",
                              lhs_val, rhs_val, viol)
    else:
        # Efficient solution: bins without periods, each bin is one "slot"
        for bin_idx, b in enumerate(solution["bins"]):
            total_w = sum(weights.get(iid, 0) for iid in b["items"])
            lhs_val = float(total_w)
            rhs_val = float(W)
            viol = lhs_val - rhs_val
            if viol > TOL:
                add_violation(violations_list, magnitudes_list, 2,
                              f"Capacity exceeded in bin {bin_idx}: "
                              f"total weight {total_w} > capacity {W}",
                              lhs_val, rhs_val, viol)

    # ------------------------------------------------------------------
    # Constraint 3 (2d): Time lag constraints (only if periods available)
    # l_{i,j} + p_i <= p_j for each arc (i,j) in A between real items
    # ------------------------------------------------------------------
    if has_periods:
        # Build period mapping for real items
        item_period = {}
        for iid, asg in assignments.items():
            if iid in real_item_ids:
                item_period[iid] = asg["period"]

        for arc in instance["arcs"]:
            i_arc = arc["from"]
            j_arc = arc["to"]
            lag = arc["lag"]

            # Only check arcs between real items
            if i_arc not in real_item_ids or j_arc not in real_item_ids:
                continue

            if i_arc not in item_period or j_arc not in item_period:
                continue

            p_i = item_period[i_arc]
            p_j = item_period[j_arc]

            # Constraint: lag + p_i <= p_j
            lhs_val = float(lag + p_i)
            rhs_val = float(p_j)
            viol = lhs_val - rhs_val  # LHS - RHS for <= constraint
            if viol > TOL:
                add_violation(violations_list, magnitudes_list, 3,
                              f"Time lag violated for arc ({i_arc}->{j_arc}): "
                              f"lag {lag} + period {p_i} = {lag + p_i} > period {p_j}",
                              lhs_val, rhs_val, viol)

    # ------------------------------------------------------------------
    # Constraint 6 (preprocessing): es(i)/ls(i) feasible time windows.
    # Source/sink arcs imply each real item's assigned period must lie in
    # [es(i), ls(i)] = [d(source, i), (T+1) - d(i, sink)] (longest paths).
    # See math_model "PREPROCESSING" section.
    # ------------------------------------------------------------------
    if has_periods:
        source_id = instance.get("source_item_id")
        sink_id = instance.get("sink_item_id")
        all_node_ids = [it["item_id"] for it in instance["items"]]

        INF = float("-inf")
        dist = {i: {j: INF for j in all_node_ids} for i in all_node_ids}
        for i in all_node_ids:
            dist[i][i] = 0
        for arc in instance["arcs"]:
            u_n, v_n, lag_n = arc["from"], arc["to"], arc["lag"]
            if lag_n > dist[u_n][v_n]:
                dist[u_n][v_n] = lag_n
        for k in all_node_ids:
            for i in all_node_ids:
                if dist[i][k] == INF:
                    continue
                for j in all_node_ids:
                    if dist[k][j] == INF:
                        continue
                    candidate = dist[i][k] + dist[k][j]
                    if candidate > dist[i][j]:
                        dist[i][j] = candidate

        es = {}
        ls = {}
        for iid in real_item_ids:
            d_si = dist[source_id][iid]
            es_i = max(1, d_si) if d_si != INF else 1
            d_if = dist[iid][sink_id]
            ls_i = (T + 1) - d_if if d_if != INF else T
            es[iid] = max(1, es_i)
            ls[iid] = min(T, ls_i)

        for iid, asg in assignments.items():
            if iid not in real_item_ids:
                continue
            p_assigned = asg["period"]
            if p_assigned < es[iid] - TOL:
                lhs_val = float(es[iid])
                rhs_val = float(p_assigned)
                viol = lhs_val - rhs_val
                add_violation(violations_list, magnitudes_list, 6,
                              f"Item {iid} assigned to period {p_assigned} < es({iid})={es[iid]}",
                              lhs_val, rhs_val, viol)
            if p_assigned > ls[iid] + TOL:
                lhs_val = float(p_assigned)
                rhs_val = float(ls[iid])
                viol = lhs_val - rhs_val
                add_violation(violations_list, magnitudes_list, 6,
                              f"Item {iid} assigned to period {p_assigned} > ls({iid})={ls[iid]}",
                              lhs_val, rhs_val, viol)

    # ------------------------------------------------------------------
    # Constraint 4 (2e): x_{i,b,p} in {0,1}
    # Implicitly satisfied by integer assignment structure.
    # Check that assignments are consistent (no fractional assignments).
    # ------------------------------------------------------------------
    if has_periods:
        for a in solution["assignments"]:
            # Each assignment represents x=1; verify no fractional values if present
            if "value" in a:
                val = a["value"]
                viol = min(abs(val - 0.0), abs(val - 1.0))
                if viol > TOL:
                    lhs_val = float(val)
                    rhs_val = round(val)
                    add_violation(violations_list, magnitudes_list, 4,
                                  f"Item {a['item_id']} assignment to bin {a['bin']}, "
                                  f"period {a['period']} is fractional: {val}",
                                  lhs_val, rhs_val, viol)

    # ------------------------------------------------------------------
    # Constraint 5 (2f): u_{b,p} in {0,1}
    # Implicitly satisfied by the solution structure (bins are either
    # active or not). Check active_bin_periods if available.
    # ------------------------------------------------------------------
    if has_periods and "active_bin_periods" in solution:
        for abp in solution["active_bin_periods"]:
            # u is implicitly 1 for active bins; nothing fractional to check
            # unless explicit values are provided
            pass

    # ------------------------------------------------------------------
    # Constraint 7: Solution-level domain/consistency of the reported
    # objective-related field `num_active_bin_periods` which MUST equal
    # the length of active_bin_periods (by the definition of the objective
    # sum_{b,p} u_{b,p}). Also validate sub-field domains (bin/period in
    # valid ranges, items non-empty and listed as integers, total_weight
    # matching the sum of member items' weights).
    # ------------------------------------------------------------------
    if has_periods:
        n_bins = instance["parameters"].get("N") or instance["parameters"].get("n_bins")
        n_periods = instance["parameters"].get("T") or instance["parameters"].get("n_periods")
        reported = solution.get("num_active_bin_periods")
        abps = solution.get("active_bin_periods", [])

        # num_active_bin_periods == len(active_bin_periods)
        if reported is not None and reported != len(abps):
            add_violation(violations_list, magnitudes_list, 7,
                          f"num_active_bin_periods={reported} does not match "
                          f"len(active_bin_periods)={len(abps)}",
                          float(reported), float(len(abps)),
                          abs(reported - len(abps)))

        # Per-abp domain checks
        for idx, abp in enumerate(abps):
            b = abp.get("bin")
            p = abp.get("period")
            items = abp.get("items", [])
            tw = abp.get("total_weight")

            # Bins are 0-indexed: 0..n_bins-1 (gurobi_code.py uses range(num_bins))
            if n_bins is not None and not (isinstance(b, int) and 0 <= b < n_bins):
                add_violation(violations_list, magnitudes_list, 7,
                              f"active_bin_periods[{idx}]: bin {b!r} out of range "
                              f"[0,{n_bins - 1}]",
                              float(b) if isinstance(b, (int, float)) else -1.0,
                              float(n_bins - 1), 1.0)
            # Periods are 1-indexed: 1..T (per math_model set T = {1,...,T})
            if n_periods is not None and not (isinstance(p, int) and 1 <= p <= n_periods):
                add_violation(violations_list, magnitudes_list, 7,
                              f"active_bin_periods[{idx}]: period {p!r} out of range "
                              f"[1,{n_periods}]",
                              float(p) if isinstance(p, (int, float)) else -1.0,
                              float(n_periods), 1.0)

            # items non-empty (since entry represents an ACTIVE bin-period)
            if not items:
                add_violation(violations_list, magnitudes_list, 7,
                              f"active_bin_periods[{idx}]: items list empty for "
                              f"an 'active' bin-period pair (bin={b}, period={p})",
                              0.0, 1.0, 1.0)

            # total_weight == sum of member items' weights (cross-check with instance)
            if tw is not None and items:
                expected_w = 0
                all_known = True
                for iid in items:
                    if iid in weights:
                        expected_w += weights[iid]
                    else:
                        all_known = False
                        break
                if all_known and int(tw) != int(expected_w):
                    add_violation(violations_list, magnitudes_list, 7,
                                  f"active_bin_periods[{idx}]: total_weight={tw} "
                                  f"does not match sum of member weights={expected_w} "
                                  f"(bin={b}, period={p})",
                                  float(tw), float(expected_w),
                                  abs(tw - expected_w))

    # ------------------------------------------------------------------
    # Constraint 8: Objective consistency (Tier C anti-gaming defense).
    # The reported objective_value MUST equal the recomputed objective (2a),
    #   min sum_{b in L} sum_{p in T} u_{b,p},
    # i.e. the number of bin-period pairs that contain at least one item.
    # This is a FULL recompute: every variable the objective depends on
    # (the item-to-(bin,period) assignment, hence which (b,p) pairs are
    # non-empty) is present in the solution -- no second-stage / scenario
    # variables are missing. For the gurobi format that count is the number
    # of distinct (bin, period) pairs over `assignments`; for the efficient
    # format (no periods) it is the number of non-empty `bins` entries.
    # An LLM candidate that fabricates objective_value (e.g. 0.0 or
    # sys.float_info.max) to game the score is rejected here.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            if has_periods:
                used_bin_periods = set(
                    (a["bin"], a["period"]) for a in solution["assignments"]
                )
                true_obj = float(len(used_bin_periods))
            else:
                true_obj = float(sum(1 for b in solution["bins"] if b.get("items")))
            abs_diff = abs(reported - true_obj)
            # Objective (2a) is an integer count of non-empty bin-periods:
            # any mismatch of >= 1 must fire.
            tol = 0.5
            if abs_diff > tol:
                add_violation(violations_list, magnitudes_list, 8,
                              f"Objective consistency violated: reported "
                              f"objective_value={reported} differs from recomputed "
                              f"objective sum_(b,p) u_(b,p)={true_obj} (number of "
                              f"non-empty bin-period pairs) "
                              f"(|diff|={abs_diff:.3g}, tol={tol})",
                              reported, true_obj, abs_diff)

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    violated_indices = sorted(set(c for c, _ in violations_list))
    # Aggregate messages by constraint index
    messages = []
    for idx in violated_indices:
        msgs = [msg for c, msg in violations_list if c == idx]
        messages.extend(msgs)

    feasible = len(violated_indices) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": messages,
        "violation_magnitudes": magnitudes_list if not feasible else [],
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for BPPTL (Letelier et al., 2022)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraint violations found.")
    else:
        print(f"INFEASIBLE - {len(result['violated_constraints'])} constraint(s) violated:")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
