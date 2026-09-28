def feasibility_check(data, solution):
    """
    Checks the feasibility of a solution for the Graph Coloring problem.
    Returns a standardized result dictionary.
    """
    # Constants for tolerances
    TOL = 1e-5
    OBJ_TOL = 0.5
    
    # Initialize result dictionary structure
    res = {
        "feasible": False,
        "objective_value": None,
        "status": "",
        "warnings": [],
        "violated_constraints": [],
        "violations": [],
        "violation_magnitudes": []
    }

    try:
        # Extract graph data
        nodes = data['graph']['nodes']
        edges = data['graph']['edges']
    except Exception:
        res["status"] = "MALFORMED_SOLUTION"
        res["warnings"].append("Input data missing required graph sections.")
        return res

    # --- PHASE 1: SYNTAX AND STRUCTURE CHECK ---
    
    # Check if solution itself is a dictionary/object
    if not isinstance(solution, dict):
        res["status"] = "MALFORMED_SOLUTION"
        res["warnings"].append(f"Solutions must be objects (dict), got {type(solution).__name__}.")
        return res

    # Determine if this is a partial solution fragment
    is_partial = solution.get("__partial__", False)

    # Validate vertex_colors
    vc = solution.get("vertex_colors")
    if not isinstance(vc, dict):
        res["status"] = "MALFORMED_SOLUTION"
        res["warnings"].append("'vertex_colors' field is missing or not a dictionary.")
        return res

    # Helper to retrieve color safely allowing for mixed int/str keys
    def get_color_raw(nid, raw_vc):
        if nid in raw_vc:
            return raw_vc[nid]
        if str(nid) in raw_vc:
            return raw_vc[str(nid)]
        return None

    # Build normalized color map and validate types
    color_map = {}
    missing_nodes = []
    invalid_values = []
    
    # Determine nodes to check (all or subset based on partial flag)
    nodes_to_check = nodes
    
    for n in nodes:
        val = get_color_raw(n, vc)
        if val is None:
            if not is_partial:
                missing_nodes.append(n)
            continue
        
        # Normalize value to integer for consistency
        norm_val = None
        try:
            if isinstance(val, float):
                if abs(val - round(val)) > TOL:
                    invalid_values.append((n, val))
                    continue
                norm_val = int(round(val))
            elif isinstance(val, int):
                norm_val = val
            else:
                invalid_values.append((n, val))
                continue
            
            color_map[n] = norm_val
        except ValueError:
            invalid_values.append((n, val))
            continue

    if missing_nodes:
        res["status"] = "MALFORMED_SOLUTION"
        res["warnings"].append(f"Vertex colors missing for vertices: {missing_nodes}.")
        return res

    if invalid_values:
        res["status"] = "MALFORMED_SOLUTION"
        msg_parts = [f"{nid}: {val}" for nid, val in invalid_values[:5]]
        if len(invalid_values) > 5:
            msg_parts.append(...)
        res["warnings"].append(f"Invalid color value types detected: {', '.join(msg_parts)}")
        return res

    # --- PHASE 2: COMPUTE OBJECTIVE VALUE ---
    unique_colors = set(color_map.values())
    computed_obj = len(unique_colors)
    
    # Assign recomputed objective value (always set, even if partial/infeasible per instructions)
    # Note: For fragments (partials), the instruction says "Return 'objective_value' as None for a fragment" 
    # but also "ALWAYS return your own recomputed objective".
    # The partial-specific instruction "Return 'objective_value' as None" takes precedence for partials.
    if is_partial:
        res["objective_value"] = None
    else:
        res["objective_value"] = computed_obj

    # --- PHASE 3: CONSISTENCY CHECK (Objective Value) ---
    if not is_partial:
        # If not partial, we typically expect objective_value to be present for consistency check
        # However, we only check it IF it is present. 
        # If missing and it's a full solution, it might be a structural issue.
        # Prompt: "If the solution contains an 'objective_value' field, you MUST verify..."
        reported_obj = solution.get("objective_value")
        
        # If missing entirely in non-partial, warn but do not necessarily fail yet (though often it is required)
        # To be safe against "malformed": Let's treat missing objective_value in a full solution as Malformed
        # because it's in the JSON schema required fields.
        if reported_obj is None:
             res["status"] = "MALFORMED_SOLUTION"
             res["warnings"].append("Missing required field 'objective_value' for full solution.")
             return res
        
        if abs(float(computed_obj) - float(reported_obj)) > OBJ_TOL:
            res["status"] = "INCONSISTENT_OBJECTIVE"
            res["warnings"].append(f"Solution is valid, but the reported objective_value {float(reported_obj)} is inconsistent with the recomputed objective {float(computed_obj)}.")
            res["objective_value"] = computed_obj
            # For consistency error, we consider it technically feasible structurally but reported badly.
            # However, prompt says set "feasible" to False.
            res["feasible"] = False
            # Add a violation record? The instructions say for INCONSISTENT_OBJECTIVE:
            # "Do NOT list any rule as violated in this case."
            # But we still fill the structure.
            # Let's keep violations empty as per instructions.
            return res

    # --- PHASE 4: CONSTRAINT CHECKING (Adjacency) ---
    violated_edges = []
    
    for i, (u, v) in enumerate(edges):
        c_u = color_map.get(u)
        c_v = color_map.get(v)
        
        # We check if both have assigned colors (for partials, some edges won't have both)
        if c_u is not None and c_v is not None:
            if c_u == c_v:
                violated_edges.append({"idx": i, "u": u, "v": v, "c": c_u})

    if violated_edges:
        res["feasible"] = False
        if is_partial:
            res["status"] = "PARTIAL_INFEASIBLE"
            res["objective_value"] = None
        else:
            res["status"] = "INFEASIBLE"
            res["objective_value"] = computed_obj
            
        for v_info in violated_edges:
            # Add constraint indices (all mapped to constraint 1 for coloring constraints generally)
            res["violated_constraints"].append(1)
            
            u_idx, v_idx, col_val = v_info["u"], v_info["v"], v_info["c"]
            
            # Format violation message
            msg = f"Constraint 1 (vertices connected by an edge cannot share a color), ({u_idx}, {v_idx}): computed color {col_val} equals {col_val}, required different, off by 1.0"
            res["violations"].append(msg)
            
            mag = {
                "constraint": 1,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            }
            res["violation_magnitudes"].append(mag)
            
        return res

    # --- PHASE 5: FINAL STATUS ---
    if is_partial:
        res["status"] = "PARTIAL_FEASIBLE"
    else:
        res["status"] = "OK"
        
    res["feasible"] = True
    # Objective value is already set correctly based on phase 2
    return res