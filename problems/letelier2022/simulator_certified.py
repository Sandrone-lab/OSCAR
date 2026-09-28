from collections import Counter

def feasibility_check(data, solution):
    TOL = 1e-5
    warnings_list = []
    violated_constraints = []
    violation_diagnose = []
    violations = []
    violation_magnitudes = []

    try:
        if not isinstance(solution, dict):
            raise ValueError("Root solution is not a dictionary.")
            
        is_partial = solution.get("__partial__") is True
        
        req_asgn_keys = {"item_id", "bin", "period", "weight"}
        
        if "assignments" not in solution:
            raise ValueError("Missing required top-level field 'assignments'.")
        assignments = solution["assignments"]
        if not isinstance(assignments, list):
            raise ValueError("'assignments' must be a list.")
        if len(assignments) == 0:
            raise ValueError("'assignments' is an empty list; at least one assignment is required.")
            
        for i, entry in enumerate(assignments):
            if not isinstance(entry, dict):
                raise ValueError(f"Assignment {i} is not a dictionary.")
            if not req_asgn_keys.issubset(entry.keys()):
                raise ValueError(f"Assignment {i} missing required fields: {req_asgn_keys - entry.keys()}.")
            for k in req_asgn_keys:
                if not isinstance(entry[k], (int, float)):
                    raise ValueError(f"Assignment {i}: '{k}' must be numeric.")

        if not is_partial:
            for rfield in ["num_active_bin_periods", "active_bin_periods"]:
                if rfield not in solution:
                    raise ValueError(f"Missing required top-level field '{rfield}'.")
            abp = solution["active_bin_periods"]
            if not isinstance(abp, list):
                raise ValueError("'active_bin_periods' must be a list.")
                
    except ValueError as ve:
        return {
            "feasible": False,
            "objective_value": None,
            "status": "MALFORMED_SOLUTION",
            "warnings": [],
            "violated_constraints": [],
            "violation_diagnose": [str(ve)],
            "violations": [f"MALFORMED_SOLUTION: {str(ve)}"],
            "violation_magnitudes": []
        }

    params = data.get("parameters", {})
    W = params.get("W")
    T = params.get("T")
    N = params.get("N")
    items_data = data.get("items", [])
    arcs_data = data.get("arcs", [])
    source_id = data.get("source_item_id")
    sink_id = data.get("sink_item_id")

    item_weights = {}
    for it in items_data:
        if it.get("type") == "treatment":
            item_weights[it["item_id"]] = float(it["weight"])
            
    precedences = [(arc["from"], arc["to"], int(arc["lag"])) for arc in arcs_data]
    
    time_map = {}
    current_assignments = {}
    for asgn in assignments:
        iid = int(asgn["item_id"])
        p = int(asgn["period"])
        current_assignments[iid] = asgn
        time_map[iid] = p
        
    if source_id is not None: 
        time_map[int(source_id)] = 0
    if sink_id is not None: 
        time_map[int(sink_id)] = T + 1

    def add_violation(idx, lhs, rhs, excess, diag="", msg=""):
        violated_constraints.append(idx)
        violation_diagnose.append(diag)
        violations.append(msg)
        violation_magnitudes.append({
            "constraint": idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(excess),
            "normalizer": 1.0,
            "ratio": float(excess)
        })

    # Rule 1: Weight Match
    for asgn in assignments:
        iid = int(asgn["item_id"])
        w = float(asgn["weight"])
        if iid not in item_weights:
            msg = f"Constraint 1 (Weight of item {iid} must equal declared weight), item {iid}: computed {w} against N/A, off by {w}"
            add_violation(1, w, 0.0, w, diag=f"Item {iid} is not a valid treatment item in data", msg=msg)
        elif abs(w - item_weights[iid]) > TOL:
            diff = abs(w - item_weights[iid])
            msg = f"Constraint 1 (Weight of item {iid} must equal declared weight), item {iid}: computed {w} against {item_weights[iid]}, off by {diff}"
            add_violation(1, w, item_weights[iid], diff, diag=f"Weight mismatch for item {iid}", msg=msg)

    # Rule 2: Period Bounds
    for asgn in assignments:
        iid = int(asgn["item_id"])
        p = int(asgn["period"])
        if p < 1 or p > T:
            dist = max(0, 1 - p) + max(0, p - T)
            msg = f"Constraint 2 (Period for item {iid} must be within [1, {T}]), item {iid}: computed {p} against [{1}, {T}], off by {dist}"
            add_violation(2, float(p), float(T), dist, diag=f"Period {p} out of legal bounds for item {iid}", msg=msg)

    # Rule 3: Precedence Lag
    for u, v, lag in precedences:
        tu = time_map.get(int(u))
        tv = time_map.get(int(v))
        if tu is not None and tv is not None:
            req = tu + lag
            if tv < req - TOL:
                diff = req - tv
                msg = f"Constraint 3 (Arc from {u} to {v} with lag {lag} requires period[{v}] >= period[{u}] + {lag}), pair ({u},{v}): computed {tv} against {req}, off by {diff}"
                add_violation(3, float(tv), float(req), diff, diag=f"Lag constraint violated for arc ({u},{v})", msg=msg)

    # Rule 4: Bin-Period Capacity
    loads = {}
    for asgn in assignments:
        bp = (int(asgn["bin"]), int(asgn["period"]))
        loads[bp] = loads.get(bp, 0.0) + float(asgn["weight"])

    for bp, load in loads.items():
        b, p = bp
        if load > W + TOL:
            diff = load - W
            msg = f"Constraint 4 (Total weight in bin {b} period {p} must not exceed capacity {W}), bin-period ({b},{p}): computed {load} against {W}, off by {diff}"
            add_violation(4, load, float(W), diff, diag=f"Capacity exceeded in bin {b} period {p}", msg=msg)

    obj_val = 0
    used_bps_from_abp = set()
    expected_used_bps = set(loads.keys())
    
    if not is_partial:
        # Rule 5: Exact Assignment
        counts = Counter(int(a["item_id"]) for a in assignments)
        for iid in item_weights:
            c = counts.get(iid, 0)
            if c != 1:
                diff = abs(c - 1)
                msg = f"Constraint 5 (Each treatment item {iid} must be assigned exactly once), item {iid}: computed {c} assignments against 1, off by {diff}"
                add_violation(5, float(c), 1.0, diff, diag=f"Incorrect assignment count for item {iid}", msg=msg)

        # Rule 6: Active Bin Period Validity
        abp_list = solution["active_bin_periods"]
        for idx, abp_entry in enumerate(abp_list):
            b = int(abp_entry["bin"])
            p = int(abp_entry["period"])
            items_in_abp = abp_entry.get("items", [])
            total_w_abp = float(abp_entry.get("total_weight", 0))
            
            if N is not None and (b < 0 or b >= N):
                dist = max(0, -b) + max(0, b - (N - 1))
                msg = f"Constraint 6 (Active bin-period bin index must be in [0, {N-1}]), entry {idx}: computed {b} against [{0}, {N-1}], off by {dist}"
                add_violation(6, float(b), float(N-1), dist, diag=f"Invalid bin index {b} in active_bin_periods entry {idx}", msg=msg)
                
            if p < 1 or p > T:
                dist = max(0, 1 - p) + max(0, p - T)
                msg = f"Constraint 6 (Active bin-period period must be in [1, {T}]), entry {idx}: computed {p} against [{1}, {T}], off by {dist}"
                add_violation(6, float(p), float(T), dist, diag=f"Invalid period {p} in active_bin_periods entry {idx}", msg=msg)
                
            used_bps_from_abp.add((b, p))
            
            expected_items = [int(a["item_id"]) for a in assignments if int(a["bin"]) == b and int(a["period"]) == p]
            if sorted(items_in_abp) != sorted(expected_items):
                diff = len(set(items_in_abp) ^ set(expected_items))
                msg = f"Constraint 6 (Items in active bin-period must match assignment), entry {idx}: computed {len(items_in_abp)} items against {len(expected_items)}, off by {diff}"
                add_violation(6, len(items_in_abp), len(expected_items), diff, diag=f"Item list mismatch in active_bin_periods entry {idx}", msg=msg)
            
            computed_total = sum(float(current_assignments[i]["weight"]) for i in expected_items) if expected_items else 0.0
            if abs(total_w_abp - computed_total) > TOL:
                diff = abs(total_w_abp - computed_total)
                msg = f"Constraint 6 (Total weight in active bin-period must match sum of assigned items), entry {idx}: computed {total_w_abp} against {computed_total}, off by {diff}"
                add_violation(6, total_w_abp, computed_total, diff, diag=f"Total weight mismatch in active_bin_periods entry {idx}", msg=msg)

        if used_bps_from_abp != expected_used_bps:
            missing = expected_used_bps - used_bps_from_abp
            extra = used_bps_from_abp - expected_used_bps
            diff = abs(len(missing) + len(extra))
            msg = f"Constraint 6 (Active bin-periods must exactly match used bin-periods), set: computed size {len(used_bps_from_abp)} against {len(expected_used_bps)}, off by {diff}"
            add_violation(6, float(len(used_bps_from_abp)), float(len(expected_used_bps)), diff, diag="Active bin-periods list does not exactly cover all used bin-periods.", msg=msg)
            
        obj_val = len(used_bps_from_abp)
    else:
        obj_val = len(loads)

    inconsistent_obj = False
    if not is_partial and "objective_value" in solution:
        reported = solution["objective_value"]
        if abs(reported - obj_val) > 0.5:
            inconsistent_obj = True
            warnings_list.append(f"Solution is valid, but the reported objective_value {reported:.4f} is inconsistent with the recomputed objective {obj_val}.")

    feasible = len(violated_constraints) == 0
    
    if is_partial:
        status = "PARTIAL_FEASIBLE" if feasible else "PARTIAL_INFEASIBLE"
        final_obj = None
    else:
        if feasible and inconsistent_obj:
            status = "INCONSISTENT_OBJECTIVE"
            feasible = False
            final_obj = float(obj_val)
        elif feasible:
            status = "OK"
            final_obj = float(obj_val)
        else:
            status = "INFEASIBLE"
            final_obj = None

    return {
        "feasible": feasible,
        "objective_value": final_obj,
        "status": status,
        "warnings": warnings_list,
        "violated_constraints": violated_constraints,
        "violation_diagnose": violation_diagnose,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }