def feasibility_check(data, solution):
    TOL = 1e-5

    n = data["n"]
    m = data["m"]
    num_scenarios = data["num_scenarios"]
    capacities = data["facility_capacity_limits"]
    cap_reqs = data["capacity_requirements"]
    release_times = data["release_times"]
    scenarios = data["scenarios"]

    result = {
        "feasible": False,
        "objective_value": None,
        "status": "",
        "warnings": [],
        "violated_constraints": [],
        "violations": [],
        "violation_magnitudes": []
    }

    is_partial = isinstance(solution, dict) and solution.get("__partial__") is True

    # -------------------------------------------------------
    # 1. MALFORMED CHECKS (Syntax & Structure)
    # -------------------------------------------------------
    if not isinstance(solution, dict):
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("MALFORMED: top-level object must be a JSON object/dictionary.")
        return result

    if not is_partial:
        for req_field in ["assignment", "schedule"]:
            if req_field not in solution:
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: missing required top-level field '{req_field}'.")
                return result

    assignment_raw = solution.get("assignment", {})
    schedule_raw = solution.get("schedule", {})

    if not isinstance(assignment_raw, dict) or not isinstance(schedule_raw, dict):
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("MALFORMED: 'assignment' and 'schedule' must be dictionaries.")
        return result

    if not is_partial and len(assignment_raw) == 0:
        result["status"] = "MALFORMED_SOLUTION"
        result["violations"].append("MALFORMED: 'assignment' is empty. Must assign each job to exactly one facility.")
        return result

    parsed_assign = {}
    for k, v in assignment_raw.items():
        try:
            j = int(k)
        except Exception:
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"MALFORMED: assignment key '{k}' cannot be converted to an integer.")
            return result
        if not isinstance(v, int) or v < 0 or v >= m:
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"MALFORMED: job {j} assigned to invalid facility {v}. Must be integer in [0, {m-1}].")
            return result
        parsed_assign[str(j)] = v

    if not is_partial:
        for j in range(n):
            if str(j) not in parsed_assign:
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: job {j} missing from assignment. All jobs must be assigned.")
                return result

    parsed_sched = {}
    for scen_k, inner in schedule_raw.items():
        if not isinstance(inner, dict):
            result["status"] = "MALFORMED_SOLUTION"
            result["violations"].append(f"MALFORMED: schedule entry for scenario '{scen_k}' is not a dictionary.")
            return result

        inner_parsed = {}
        for job_k, ent in inner.items():
            if not isinstance(ent, dict):
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: job '{job_k}' in scenario '{scen_k}' is not a dictionary.")
                return result
            if not all(field in ent for field in ("facility", "start", "finish")):
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: job '{job_k}' in scenario '{scen_k}' missing required fields (facility, start, finish).")
                return result
            try:
                j = int(job_k)
            except Exception:
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: schedule job key '{job_k}' in scenario '{scen_k}' is not a valid integer.")
                return result
            fac = ent["facility"]
            st_raw = ent["start"]
            ft_raw = ent["finish"]

            if not isinstance(fac, int) or fac < 0 or fac >= m:
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: job {j} in scenario {scen_k} facility {fac} out of range [0, {m-1}].")
                return result
            try:
                st = float(st_raw)
                ft = float(ft_raw)
            except Exception:
                result["status"] = "MALFORMED_SOLUTION"
                result["violations"].append(f"MALFORMED: job {j} in scenario {scen_k} start/finish must be numeric.")
                return result

            inner_parsed[str(j)] = {"facility": fac, "start": st, "finish": ft}
        parsed_sched[scen_k] = inner_parsed

    # -------------------------------------------------------
    # 2. PARTIAL INPUT HANDLING
    # -------------------------------------------------------
    if is_partial:
        violations_list = []
        mag_list = []
        idx = 1

        for scen_k, inner in parsed_sched.items():
            for j_s, ent in inner.items():
                j = int(j_s)
                fac_sched = ent["facility"]

                fac_assign = parsed_assign.get(str(j))
                if fac_assign is not None and fac_assign != fac_sched:
                    violations_list.append(f"Constraint {idx} (assignment-schedule consistency), job {j_s} in scenario {scen_k}: assigned facility {fac_assign} != scheduled facility {fac_sched}, off by {abs(fac_assign - fac_sched)}.")
                    mag_list.append({"constraint": idx, "lhs": fac_sched, "rhs": fac_assign, "raw_excess": abs(fac_assign - fac_sched), "normalizer": max(1, m), "ratio": abs(fac_assign - fac_sched) / max(1, m)})
                    idx += 1

                rt = release_times[j]
                st = ent["start"]
                if st < rt - TOL:
                    violations_list.append(f"Constraint {idx} (release time), job {j_s}: start {st:.6f} < release time {rt}, off by {rt - st:.6f}.")
                    mag_list.append({"constraint": idx, "lhs": st, "rhs": rt, "raw_excess": rt - st, "normalizer": max(1.0, abs(float(rt))), "ratio": (rt - st) / max(1.0, abs(float(rt)))})
                    idx += 1

                try:
                    scen_idx = int(scen_k)
                    pt = float(scenarios[scen_idx]["processing_times"][fac_sched][j])
                    comp_finish = st + pt
                    ft = ent["finish"]
                    if abs(comp_finish - ft) > TOL:
                        diff = abs(comp_finish - ft)
                        norm = max(1.0, abs(ft))
                        violations_list.append(f"Constraint {idx} (processing time equality), job {j_s} on facility {fac_sched}: computed finish {comp_finish:.6f} != reported finish {ft:.6f}, off by {diff:.6f}.")
                        mag_list.append({"constraint": idx, "lhs": comp_finish, "rhs": ft, "raw_excess": diff, "normalizer": norm, "ratio": diff / norm})
                        idx += 1
                except Exception:
                    pass

        result["violated_constraints"] = [vm["constraint"] for vm in mag_list]
        result["violations"] = violations_list
        result["violation_magnitudes"] = mag_list
        if mag_list:
            result["feasible"] = False
            result["status"] = "PARTIAL_INFEASIBLE"
        else:
            result["feasible"] = True
            result["status"] = "PARTIAL_FEASIBLE"
        result["objective_value"] = None
        return result

    # -------------------------------------------------------
    # 3. STRUCTURAL & SEMANTIC CHECKS (Full Solution)
    # -------------------------------------------------------
    violations_list = []
    mag_list = []
    idx = 1

    # 3a. Schedule completeness per scenario
    for scen_idx in range(num_scenarios):
        scen_key = str(scen_idx)
        sc_inner = parsed_sched.get(scen_key, {})
        for j in range(n):
            if str(j) not in sc_inner:
                violations_list.append(f"Constraint {idx} (schedule completeness), job {j} in scenario {scen_idx}: absent from schedule.")
                mag_list.append({"constraint": idx, "lhs": 0, "rhs": 1, "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0})
                idx += 1

    # 3b. Assignment-Schedule consistency & Single constraints
    for scen_k, inner in parsed_sched.items():
        for j_s, ent in inner.items():
            j = int(j_s)
            fac_sched = ent["facility"]
            fac_assign = parsed_assign.get(str(j))
            if fac_assign is not None and fac_assign != fac_sched:
                violations_list.append(f"Constraint {idx} (assignment-schedule consistency), job {j_s} in scenario {scen_k}: assigned facility {fac_assign} != scheduled facility {fac_sched}, off by {abs(fac_assign - fac_sched)}.")
                mag_list.append({"constraint": idx, "lhs": fac_sched, "rhs": fac_assign, "raw_excess": abs(fac_assign - fac_sched), "normalizer": max(1, m), "ratio": abs(fac_assign - fac_sched) / max(1, m)})
                idx += 1

            # Release time
            rt = release_times[j]
            st = ent["start"]
            if st < rt - TOL:
                violations_list.append(f"Constraint {idx} (release time), job {j_s}: start {st:.6f} < release time {rt}, off by {rt - st:.6f}.")
                mag_list.append({"constraint": idx, "lhs": st, "rhs": rt, "raw_excess": rt - st, "normalizer": max(1.0, abs(float(rt))), "ratio": (rt - st) / max(1.0, abs(float(rt)))})
                idx += 1

            # Processing time equality
            try:
                scen_idx = int(scen_k)
                pt = float(scenarios[scen_idx]["processing_times"][fac_sched][j])
                comp_finish = st + pt
                ft = ent["finish"]
                if abs(comp_finish - ft) > TOL:
                    diff = abs(comp_finish - ft)
                    norm = max(1.0, abs(ft))
                    violations_list.append(f"Constraint {idx} (processing time equality), job {j_s} on facility {fac_sched}: computed finish {comp_finish:.6f} != reported finish {ft:.6f}, off by {diff:.6f}.")
                    mag_list.append({"constraint": idx, "lhs": comp_finish, "rhs": ft, "raw_excess": diff, "normalizer": norm, "ratio": diff / norm})
                    idx += 1
            except Exception:
                pass

    if violations_list:
        result["feasible"] = False
        result["status"] = "INFEASIBLE"
        result["violated_constraints"] = [vm["constraint"] for vm in mag_list]
        result["violations"] = violations_list
        result["violation_magnitudes"] = mag_list
        result["objective_value"] = None
        return result

    # 4. RESOURCE CAPACITY CONSTRAINTS & OBJECTIVE RECOMPUTATION
    recompute_obj = 0.0
    for scen_idx, scen_obj in enumerate(scenarios):
        prob = scen_obj["probability"]
        proc_mat = scen_obj["processing_times"]
        scen_key = str(scen_idx)
        sc_inner = parsed_sched[scen_key]

        fac_events = {}
        scen_ms = 0.0

        for j_s, ent in sc_inner.items():
            j = int(j_s)
            fac = ent["facility"]
            st = ent["start"]
            pt = float(proc_mat[fac][j])
            comp_ft = st + pt

            if comp_ft > scen_ms:
                scen_ms = comp_ft

            fac_events.setdefault(fac, []).append((st, comp_ft, cap_reqs[fac][j]))

        for fac, intervals in fac_events.items():
            bpts = set()
            for s, f, _ in intervals:
                bpts.add(s)
                bpts.add(f)
            for t in sorted(bpts):
                total_res = sum(req for s, f, req in intervals if s <= t < f - TOL)
                cap = capacities[fac]
                if total_res > cap + TOL:
                    excess = total_res - cap
                    norm = max(1.0, float(cap))
                    violations_list.append(f"Constraint {idx} (resource capacity on facility {fac}), time {t}: load {total_res:.4f} > capacity {cap}, off by {excess:.4f}.")
                    mag_list.append({"constraint": idx, "lhs": total_res, "rhs": cap, "raw_excess": excess, "normalizer": norm, "ratio": excess / norm})
                    idx += 1

        recompute_obj += prob * scen_ms

    if violations_list:
        result["feasible"] = False
        result["status"] = "INFEASIBLE"
        result["violated_constraints"] = [vm["constraint"] for vm in mag_list]
        result["violations"] = violations_list
        result["violation_magnitudes"] = mag_list
        result["objective_value"] = None
        return result

    # 5. OBJECTIVE CONSISTENCY CHECK
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        if abs(reported_obj - recompute_obj) > 0.5:
            result["feasible"] = False
            result["status"] = "INCONSISTENT_OBJECTIVE"
            result["objective_value"] = recompute_obj
            result["violations"] = [f"Solution is valid, but the reported objective_value {reported_obj:.4f} is inconsistent with the recomputed objective {recompute_obj:.4f}."]
            result["violated_constraints"] = []
            result["violation_magnitudes"] = []
            return result

    result["feasible"] = True
    result["status"] = "OK"
    result["objective_value"] = recompute_obj
    return result