import json
from collections import defaultdict

def feasibility_check(data, solution):
    """
    Checks the feasibility of a DPRPP-IC solution.
    Returns detailed result dictionary.
    """
    TOL = 1e-5
    
    # Initialize Result Dictionary
    result = {
        "feasible": False,
        "objective_value": None,
        "status": "",
        "warnings": [],
        "violated_constraints": [],
        "violation_diagnose": [],
        "violations": [],
        "violation_magnitudes": []
    }

    # Helper to normalize inputs if passed as lists wrapping the object
    def normalize(obj):
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        return obj

    # Normalize inputs
    data_norm = normalize(data)
    solution_norm = normalize(solution)
    
    # --- Step 1: Validate Inputs and Structure ---
    
    # Check Solution Object Type
    if not isinstance(solution_norm, dict):
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("Solution is not a valid JSON object.")
        result["violated_constraints"].append(0)
        result["violation_diagnose"].append("Format Error: Root element not an object")
        result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
        return result
    
    # Check Partial Flag
    is_partial = solution_norm.get("__partial__", False)
    
    # Required Fields Definitions
    required_full_fields = ["served_arcs", "tour_arcs", "objective_value", "total_profit", "total_travel_cost", "total_penalty"]
    required_partial_fields = ["served_arcs", "tour_arcs"]
    
    # Determine missing fields based on Partial flag
    required_fields = required_partial_fields if is_partial else required_full_fields
    missing_fields = [f for f in required_fields if f not in solution_norm]
    
    # 1.1 Check Missing Fields
    if missing_fields:
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append(f"Missing required fields: {', '.join(missing_fields)}")
        result["violated_constraints"].append(0)
        result["violation_diagnose"].append("Format Error: Missing Required Fields")
        result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
        return result
    
    # 1.2 Check Field Types (Strict Type Check)
    # Must be list
    if not isinstance(solution_norm["served_arcs"], list):
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("'served_arcs' is not a list.")
        result["violated_constraints"].append(0)
        result["violation_diagnose"].append("Format Error: Served Arcs must be array")
        result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
        return result
        
    if not isinstance(solution_norm["tour_arcs"], list):
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("'tour_arcs' is not a list.")
        result["violated_constraints"].append(0)
        result["violation_diagnose"].append("Format Error: Tour Arcs must be array")
        result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
        return result

    # 1.3 Check Element Types
    # Check Tour Arcs
    for i, arc in enumerate(solution_norm["tour_arcs"]):
        if not isinstance(arc, dict):
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"Tour arc index {i} is not an object.")
            result["violated_constraints"].append(0)
            result["violation_diagnose"].append("Format Error: Tour Arc Entry")
            result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
            return result
        if "from" not in arc or "to" not in arc or "count" not in arc or "cost" not in arc:
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"Tour arc index {i} missing required fields.")
            result["violated_constraints"].append(0)
            result["violation_diagnose"].append("Format Error: Tour Arc Entry Missing Fields")
            result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
            return result

    # Check Served Arcs
    for i, arc in enumerate(solution_norm["served_arcs"]):
        if not isinstance(arc, dict):
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"Served arc index {i} is not an object.")
            result["violated_constraints"].append(0)
            result["violation_diagnose"].append("Format Error: Served Arc Entry")
            result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
            return result
        if "from" not in arc or "to" not in arc or "profit" not in arc or "cost" not in arc:
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"Served arc index {i} missing required fields.")
            result["violated_constraints"].append(0)
            result["violation_diagnose"].append("Format Error: Served Arc Entry Missing Fields")
            result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
            return result
            
    # 1.4 Check Objective Value Type (if Full Solution)
    if not is_partial:
        if not isinstance(solution_norm["objective_value"], (int, float)):
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append("'objective_value' is not a number.")
            result["violated_constraints"].append(0)
            result["violation_diagnose"].append("Format Error: Objective Value Type")
            result["violation_magnitudes"].append({"constraint": 0, "lhs": 0.0, "rhs": 0.0, "raw_excess": 0.0, "normalizer": 0.0, "ratio": 0.0})
            return result

    # --- Step 2: Prepare Data Structures ---
    vi_nodes = set(data_norm.get("VI_nodes", []))
    depot = int(data_norm.get("depot", 0))
    
    weak_pen_map = {}
    for w_data in data_norm.get("weak_incompatibilities", []):
        if not isinstance(w_data, list) or len(w_data) < 2: continue
        u, v = int(w_data[0]), int(w_data[1])
        pen = float(w_data[2]) if len(w_data) > 2 else 0.0
        weak_pen_map[(min(u, v), max(u, v))] = pen 
        
    strong_pair_set = set()
    for inc in data_norm.get("strong_incompatibilities", []):
        if not isinstance(inc, list) or len(inc) < 2: continue
        u, v = int(inc[0]), int(inc[1])
        strong_pair_set.add(tuple(sorted((u, v))))
        
    adj_graph = defaultdict(set)
    flow_in = defaultdict(int)
    flow_out = defaultdict(int)
    tour_counts = defaultdict(int)
    traveled_arcs_set = set()
    travel_cost_calc = 0.0
    
    active_vi_nodes = set()
    total_profit_calc = 0.0
    
    # Process Tour Arcs
    for arc in solution_norm["tour_arcs"]:
        u, v = int(arc["from"]), int(arc["to"])
        count = int(arc["count"])
        cost = float(arc["cost"])
        
        if count > 0:
            tour_counts[(u, v)] += count
            traveled_arcs_set.add((u, v))
            adj_graph[u].add(v)
            flow_out[u] += count
            flow_in[v] += count
            travel_cost_calc += count * cost
            
    # Process Served Arcs
    for arc in solution_norm["served_arcs"]:
        u, v = int(arc["from"]), int(arc["to"])
        profit = float(arc["profit"])
        total_profit_calc += profit
        if u in vi_nodes:
            active_vi_nodes.add(u)

    # --- Step 3: Rule Checks ---
    violations_list = []
    diagnoses_list = []
    mag_list = []
    violated = False
    current_status = "OK" if not is_partial else "PARTIAL_FEASIBLE"
    
    # 1. Flow Conservation
    flow_violated = False
    flow_nodes_violated = []
    all_involved_nodes = set(flow_in.keys()) | set(flow_out.keys())
    
    for n in all_involved_nodes:
        if abs(flow_in[n] - flow_out[n]) > TOL:
            flow_violated = True
            flow_nodes_violated.append(n)
            diff = abs(flow_in[n] - flow_out[n])
            violations_list.append(f"Constraint 1 (Flow Conservation), node {n}: computed in-flow={flow_in[n]}, out-flow={flow_out[n]}, off by {diff}")
            diagnoses_list.append("Flow conservation violated: in-degree != out-degree")
            mag_list.append({
                "constraint": 1,
                "lhs": float(diff),
                "rhs": 0.0,
                "raw_excess": diff,
                "normalizer": 1.0,
                "ratio": diff
            })
            for _ in range(1):
                 result["violated_constraints"].append(1)
    
    if flow_violated:
        violated = True
        current_status = "INFEASIBLE" if not is_partial else "PARTIAL_INFEASIBLE"

    # 2. Connectivity
    conn_violated = False
    if flow_violated == False and len(tour_counts) > 0:
        visited = set([depot])
        queue = [depot]
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            if curr in adj_graph:
                for neigh in adj_graph[curr]:
                    if neigh not in visited:
                        visited.add(neigh)
                        queue.append(neigh)
        
        involved = set(flow_in.keys()) | set(flow_out.keys())
        if not involved.issubset(visited):
            conn_violated = True
            violations_list.append(f"Constraint 2 (Connectivity), nodes involved: {sorted(list(involved))}: disconnected tour found")
            diagnoses_list.append("Tour connectivity violated: nodes not reachable from depot")
            mag_list.append({
                "constraint": 2,
                "lhs": 1.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })
            result["violated_constraints"].append(2)
        elif depot not in involved and len(involved) > 0:
            conn_violated = True
            violations_list.append(f"Constraint 2 (Connectivity), nodes involved: {sorted(list(involved))}: tour not starting/ending at depot")
            diagnoses_list.append("Tour connectivity violated: does not start/end at depot")
            mag_list.append({
                "constraint": 2,
                "lhs": 1.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })
            result["violated_constraints"].append(2)

    if conn_violated:
        violated = True
        current_status = "INFEASIBLE" if not is_partial else "PARTIAL_INFEASIBLE"

    # 3. Service Coverage
    service_violated = False
    for arc in solution_norm["served_arcs"]:
        u, v = int(arc["from"]), int(arc["to"])
        if tour_counts[(u, v)] < 1:
            service_violated = True
            violations_list.append(f"Constraint 3 (Service Coverage), served arc ({u},{v}): traversed count=0")
            diagnoses_list.append("Service coverage violated: served arc not traversed")
            mag_list.append({
                "constraint": 3,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })
            result["violated_constraints"].append(3)
            break
            
    if service_violated:
        violated = True
        current_status = "INFEASIBLE" if not is_partial else "PARTIAL_INFEASIBLE"

    # 4. Strong Incompatibility
    strong_violated = False
    for pkey in strong_pair_set:
        u, v = pkey
        if u in active_vi_nodes and v in active_vi_nodes:
            strong_violated = True
            violations_list.append(f"Constraint 4 (Strong Incompatibility), nodes {u}, {v}: both active but strongly incompatible")
            diagnoses_list.append("Strong incompatibility violated: mutually exclusive nodes both active")
            mag_list.append({
                "constraint": 4,
                "lhs": 1.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })
            result["violated_constraints"].append(4)
            break
            
    if strong_violated:
        violated = True
        current_status = "INFEASIBLE" if not is_partial else "PARTIAL_INFEASIBLE"
        
    # Append collected violations to result
    result["violations"].extend(violations_list)
    result["violation_diagnose"].extend(diagnoses_list)
    result["violation_magnitudes"].extend(mag_list)
    
    # Recompute Objective
    recalc_penalty = 0.0
    seen_weak_keys = set()
    for u in active_vi_nodes:
        for v in active_vi_nodes:
            if u >= v: continue
            k = (u, v)
            if k in weak_pen_map:
                if k not in seen_weak_keys:
                    recalc_penalty += weak_pen_map[k]
                    seen_weak_keys.add(k)
    
    recalc_obj = total_profit_calc - travel_cost_calc - recalc_penalty
    result["objective_value"] = recalc_obj  # Always store computed value

    if violated:
        result["feasible"] = False
        result["status"] = current_status
        return result

    # --- Step 4: Objective Consistency Check (Full Solution Only) ---
    rep_obj = float(solution_norm["objective_value"])
    
    if not is_partial:
        diff = abs(rep_obj - recalc_obj)
        if diff > 0.5:
            result["feasible"] = False
            result["status"] = "INCONSISTENT_OBJECTIVE"
            # Reset violations as instructed for this status
            result["violated_constraints"] = []
            result["violations"] = []
            result["violation_diagnose"] = []
            result["violation_magnitudes"] = []
            result["warnings"].append(f"Inconsistent objective: reported {rep_obj}, recomputed {recalc_obj}")
            return result

    # --- Step 5: Feasible ---
    result["feasible"] = True
    result["status"] = "OK"
    return result