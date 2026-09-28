def feasibility_check(data, solution):
    TOL = 1e-5
    
    result = {
        "feasible": False,
        "objective_value": None,
        "status": "INFEASIBLE",
        "warnings": [],
        "violated_constraints": [],
        "violations": [],
        "violation_magnitudes": []
    }
    
    # PHASE 0: Basic shape validation
    if not isinstance(solution, dict):
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append(f"Solution is not a JSON object ({type(solution).__name__}). Must be a dict.")
        return result
        
    is_partial = bool(solution.get("__partial__", False))
    
    has_arcs = "solution_arcs" in solution
    has_x = "solution_x" in solution
    
    if not has_arcs and not has_x:
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("Missing path representation: solution must contain either 'solution_arcs' or 'solution_x'.")
        return result
        
    if has_arcs and has_x:
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("Duplicate path representations: provided both 'solution_arcs' and 'solution_x'. Use exactly one.")
        return result
        
    selected_ids = set()
    
    if has_arcs:
        raw = solution["solution_arcs"]
        if not isinstance(raw, list):
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append("'solution_arcs' is not a list.")
            return result
            
        if len(raw) == 0 and not is_partial:
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append("'solution_arcs' is empty; at least one arc must be selected.")
            return result
            
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"'solution_arcs[{idx}]' is not a dict.")
                return result
                
            for k in ("id", "from_node", "to_node"):
                if k not in entry:
                    result["status"] = "MALFORMED_SOLUTION"
                    result["violations"].append(f"'solution_arcs[{idx}]' missing required key '{k}'.")
                    return result
                if not isinstance(entry[k], int):
                    result["status"] = "MALFORMED_SOLUTION"
                    result["violations"].append(f"'solution_arcs[{idx}]['{k}'] is not an integer.")
                    return result
                    
            aid = entry["id"]
            if aid in selected_ids:
                result["status"] = "INFEASIBLE" if not is_partial else "PARTIAL_INFEASIBLE"
                result["violations"].append(f"Rule (binary arc usage): Arc {aid} appears more than once in 'solution_arcs'.")
                result["violated_constraints"] = [-1]
                result["violation_magnitudes"] = [{"constraint": -1, "lhs": 2.0, "rhs": 1.0, "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0}]
                return result
                
            selected_ids.add(aid)
            
    else: # has_x
        xdict = solution["solution_x"]
        if not isinstance(xdict, dict):
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append("'solution_x' is not a dict.")
            return result
            
        if len(xdict) == 0 and not is_partial:
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append("'solution_x' is empty; at least one arc must be selected.")
            return result
            
        for k, v in xdict.items():
            try:
                aid = int(k)
            except Exception:
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"'solution_x' key '{k}' cannot be cast to integer.")
                return result
                
            if not isinstance(v, (int, float)):
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"'solution_x[{aid}]' is not numeric.")
                return result
                
            if abs(v - 1.0) > TOL:
                result["status"] = "INFEASIBLE" if not is_partial else "PARTIAL_INFEASIBLE"
                result["violations"].append(f"Rule (binary arc usage): solution_x[{aid}] = {v}, expected 1.0.")
                result["violated_constraints"] = [-1]
                result["violation_magnitudes"] = [{"constraint": -1, "lhs": float(v), "rhs": 1.0, "raw_excess": abs(v-1.0), "normalizer": 1.0, "ratio": abs(v-1.0)}]
                return result
                
            selected_ids.add(aid)
            
    # Handle partial case early
    if is_partial:
        result["feasible"] = True
        result["status"] = "PARTIAL_FEASIBLE"
        result["objective_value"] = None
        return result
        
    # Validate against data
    source = data["source_node"]
    target = data["target_node"]
    arc_map = {a["id"]: a for a in data["arcs"]}
    
    bad_ids = sorted([aid for aid in selected_ids if aid not in arc_map])
    if bad_ids:
        result["status"] = "INFEASIBLE"
        result["violations"].append(f"Selected arcs {bad_ids} are not in the problem definition.")
        result["violated_constraints"] = [-1]
        result["violation_magnitudes"] = [{"constraint": -1, "lhs": float(len(bad_ids)), "rhs": 0.0, "raw_excess": float(len(bad_ids)), "normalizer": 1.0, "ratio": float(len(bad_ids))}]
        return result
        
    # Degree checks
    deg_in = {}
    deg_out = {}
    for aid in selected_ids:
        fn = arc_map[aid]["from_node"]
        tn = arc_map[aid]["to_node"]
        deg_out[fn] = deg_out.get(fn, 0) + 1
        deg_in[tn] = deg_in.get(tn, 0) + 1
        
    all_nodes = set()
    for a in data["arcs"]:
        all_nodes.add(a["from_node"])
        all_nodes.add(a["to_node"])
        
    constraints_violated = []
    
    s_out = deg_out.get(source, 0)
    if s_out != 1:
        constraints_violated.append((0, f"Source node {source} out-degree must be 1", s_out, 1))
        
    t_in = deg_in.get(target, 0)
    if t_in != 1:
        constraints_violated.append((1, f"Target node {target} in-degree must be 1", t_in, 1))
        
    cnt = 2
    for n in sorted(all_nodes):
        if n in (source, target): continue
        oi = deg_out.get(n, 0)
        ii = deg_in.get(n, 0)
        if oi != ii:
            constraints_violated.append((cnt, f"Intermediate node {n} flow conservation (out={oi} vs in={ii})", oi, ii))
            cnt += 1
            
    if constraints_violated:
        vid, vmsg, vmag = [], [], []
        for cidx, desc, lhs, rhs in constraints_violated:
            excess = abs(lhs - rhs)
            vid.append(cidx)
            vmsg.append(f"Constraint {cidx} ({desc}): computed LHS {lhs} against RHS {rhs}, off by {round(excess, 6)}")
            norm = max(abs(rhs), 1.0) if abs(rhs) > TOL else 1.0
            vmag.append({"constraint": cidx, "lhs": float(lhs), "rhs": float(rhs), "raw_excess": float(excess), "normalizer": 1.0, "ratio": excess / norm})
        result["violated_constraints"] = vid
        result["violations"] = vmsg
        result["violation_magnitudes"] = vmag
        result["status"] = "INFEASIBLE"
        return result
        
    # Connectivity check
    visited = set()
    cur = source
    path_edges = []
    while True:
        if cur in visited:
            result["status"] = "INFEASIBLE"
            result["violations"].append(f"Cyclic detour discovered starting from {cur}; not a simple path.")
            result["violated_constraints"] = [-1]
            result["violation_magnitudes"] = [{"constraint": -1, "lhs": -1.0, "rhs": 0.0, "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0}]
            return result
        visited.add(cur)
        cand = [a for a in selected_ids if arc_map[a]["from_node"] == cur]
        if not cand:
            result["status"] = "INFEASIBLE"
            result["violations"].append(f"Dead-end reached at node {cur}; no selected outgoing arc found.")
            result["violated_constraints"] = [-1]
            result["violation_magnitudes"] = [{"constraint": -1, "lhs": -1.0, "rhs": 0.0, "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0}]
            return result
        arc_next = cand[0]
        path_edges.append(arc_next)
        cur = arc_map[arc_next]["to_node"]
        if cur == target:
            break
            
    if set(path_edges) != selected_ids:
        extra = sorted(selected_ids - set(path_edges))
        result["status"] = "INFEASIBLE"
        result["violations"].append(f"{len(extra)} extra selected arc(s) not reachable on the source→target path: {extra}.")
        result["violated_constraints"] = [-1]
        result["violation_magnitudes"] = [{"constraint": -1, "lhs": float(len(extra)), "rhs": 0.0, "raw_excess": float(len(extra)), "normalizer": 1.0, "ratio": float(len(extra))}]
        return result
        
    # Compute objective
    linear_sum = sum(data["linear_costs"][a] for a in selected_ids)
    quad_sum = 0.0
    sel_sorted = sorted(selected_ids)
    for a in sel_sorted:
        for b in sel_sorted:
            quad_sum += float(data["quadratic_costs"][a][b])
    true_obj = float(linear_sum + quad_sum)
    result["objective_value"] = true_obj
    
    # Check reported objective
    rep_obj = solution.get("objective_value")
    if rep_obj is not None:
        mismatch = abs(rep_obj - true_obj)
        if mismatch > 0.5:
            result["feasible"] = False
            result["status"] = "INCONSISTENT_OBJECTIVE"
            result["violations"] = [f"Solution is structurally valid, but the reported objective_value {rep_obj} is inconsistent with the recomputed objective {true_obj} (difference {mismatch})."]
            result["violated_constraints"] = []
            result["violation_magnitudes"] = []
            return result
            
    result["feasible"] = True
    result["status"] = "OK"
    return result