import sys
from collections import defaultdict
from typing import Dict, List, Any

def feasibility_check(data: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the feasibility of a schedule solution for an MMRCPSP instance.
    """
    TOL = 1e-5
    
    # Storage for violations mapped to constraint index -> (msg, lhs, rhs)
    violations_dict = {}
    
    def log_violation(idx, msg, lhs, rhs):
        if idx not in violations_dict:
            violations_dict[idx] = (msg, lhs, rhs)
        else:
            curr = violations_dict[idx]
            if (lhs - rhs) > (curr[1] - curr[2]):
                violations_dict[idx] = (msg, lhs, rhs)

    try:
        # Parse Data Structures
        jobs_map = {j["job_id"]: j for j in data["jobs"]}
        projects_map = {p["project_id"]: p for p in data["projects"]}
        
        sched_entries = solution.get("schedule", [])
        sched_map = {}
        dup_scheduled = set()
        
        # --- Constraint 0: Schedule Integrity ---
        # Check 1: Duplicates
        for entry in sched_entries:
            j_id = entry["job_id"]
            if j_id in sched_map:
                dup_scheduled.add(j_id)
            else:
                sched_map[j_id] = entry
                
        # Check 2: Completeness (All jobs in data must be in solution)
        all_jobs_present = True
        for j_id in jobs_map:
            if j_id not in sched_map:
                all_jobs_present = False
                break
        
        # Check 3: Exact count (No extra jobs)
        if len(sched_entries) != len(jobs_map):
             all_jobs_present = False

        if not all_jobs_present or dup_scheduled:
            msg = "Schedule Integrity Error: Duplicate, Missing, or Extra jobs"
            log_violation(0, msg, 1.0, 0.0)

        # --- Constraint 1: Valid Mode Selection ---
        for j_id, entry in sched_map.items():
            if j_id not in jobs_map: continue
            job = jobs_map[j_id]
            mode_id = entry["mode_id"]
            valid_mode_ids = [m["mode_id"] for m in job["modes"]]
            if mode_id not in valid_mode_ids:
                log_violation(1, f"Invalid mode {mode_id} for job {j_id}", 0.0, 1.0)
        
        # --- Prepare Job Attributes ---
        job_attr = {}
        for j_id, entry in sched_map.items():
            if j_id not in jobs_map: continue
            job = jobs_map[j_id]
            
            # Find selected mode details
            mode = None
            for m in job["modes"]:
                if m["mode_id"] == entry["mode_id"]:
                    mode = m
                    break
            if mode is None: continue
            
            proj_ref = projects_map.get(job["project_id"], {})
            
            job_attr[j_id] = {
                "start": entry["start_time"],
                "dur": mode["duration"],
                "renewable_cons": mode["renewable_consumption"],
                "nonrenewable_cons": mode["nonrenewable_consumption"],
                "successors": job["successors"],
                "release_date": proj_ref.get("release_date", 0),
                "project_id": job["project_id"],
                "type": job["type"]
            }

        # --- Constraint 2: Precedence Relationships ---
        for j_id, attr in job_attr.items():
            for succ_id in attr["successors"]:
                if succ_id in job_attr:
                    pred_finish = attr["start"] + attr["dur"]
                    succ_start = job_attr[succ_id]["start"]
                    if pred_finish > succ_start + TOL:
                        log_violation(2, f"Precedence Violation: {j_id} -> {succ_id}", pred_finish, succ_start)
        
        # --- Constraint 3: Release Dates ---
        for j_id, attr in job_attr.items():
            if attr["start"] < attr["release_date"] - TOL:
                log_violation(3, f"Release Date Violation: Job {j_id}", attr["release_date"], attr["start"])

        # --- Constraint 4: Renewable Resource Capacity ---
        renew_resources = data.get("resources", {}).get("renewable", [])
        num_renew = len(renew_resources)
        renew_caps = [(i, res["capacity"]) for i, res in enumerate(renew_resources)]
        
        # Calculate horizon based on actual completion times and makespan
        max_comp_time = 0
        for attr in job_attr.values():
            end = attr["start"] + attr["dur"]
            if end > max_comp_time: max_comp_time = end
            
        sol_makespan = solution.get("makespan")
        if sol_makespan is None or sol_makespan < max_comp_time:
            sol_makespan = max_comp_time
        
        # Ensure grid covers the maximum time
        horizon_limit = int(sol_makespan)
        
        usage_grid = [[0.0 for _ in range(num_renew)] for _ in range(horizon_limit + 1)]
        
        for attr in job_attr.values():
            start = int(attr["start"])
            dur = int(attr["dur"])
            cons = attr["renewable_cons"]
            
            # Process periods [start, start + dur)
            # Clamp to horizon limit to avoid index out of bounds (though consistency check should prevent overflow)
            end = min(start + dur, horizon_limit + 1)
            
            # Safety check for negative start or duration
            if start < 0 or dur < 0:
                continue

            for t in range(start, end):
                for r in range(num_renew):
                    usage_grid[t][r] += cons[r]
                    
        for r_idx, cap in renew_caps:
            for t in range(len(usage_grid)):
                used = usage_grid[t][r_idx]
                if used > cap + TOL:
                    log_violation(4, f"Renewable Resource {r_idx} Exceeded at t={t}", used, cap)

        # --- Constraint 5: Non-Renewable Resource Capacity ---
        nonrenew_resources = data.get("resources", {}).get("nonrenewable", [])
        num_nonrenew = len(nonrenew_resources)
        nonrenew_caps = [(i, res["capacity"]) for i, res in enumerate(nonrenew_resources)]
        
        total_nr_usage = defaultdict(float)
        for attr in job_attr.values():
            cons_list = attr["nonrenewable_cons"]
            # Ensure consumption list matches number of resources
            for i in range(min(len(cons_list), num_nonrenew)):
                total_nr_usage[i] += cons_list[i]
                
        for r_idx, cap in nonrenew_caps:
            used = total_nr_usage.get(r_idx, 0.0)
            if used > cap + TOL:
                log_violation(5, f"Non-Renewable Resource {r_idx} Capacity Exceeded", used, cap)

        # --- Constraint 6: Makespan Consistency ---
        max_completion = 0
        for attr in job_attr.values():
            end = attr["start"] + attr["dur"]
            if end > max_completion: max_completion = end
            
        if sol_makespan < max_completion - TOL:
            log_violation(6, "Makespan Inconsistency", sol_makespan, max_completion)

        # --- Constraint 7: Objective Value Consistency ---
        # Compute Critical Path Duration (CPD) for each project to verify TPD
        # Use iterative DP to avoid recursion depth issues
        project_cpds = {}
        projs = data.get("projects", [])
        
        for proj in projs:
            pid = proj["project_id"]
            sink_id = proj["artificial_sink_job_id"]
            src_id = proj["artificial_source_job_id"]
            rel_date = proj.get("release_date", 0)
            
            # Filter jobs belonging to this project
            proj_jobs = {}
            # Store min duration for CPD calc
            for j_id, job in jobs_map.items():
                if job["project_id"] == pid:
                    # Get min duration from available modes
                    if "modes" in job and len(job["modes"]) > 0:
                        min_dur = min([m["duration"] for m in job["modes"]])
                    else:
                        min_dur = 0
                    proj_jobs[j_id] = min_dur
            
            # Build adjacency list (Predecessors)
            pred_map = defaultdict(list)
            for j_id in proj_jobs:
                if j_id in job_attr: # Only consider scheduled jobs for dependency logic? 
                                     # Actually CPD is structural, independent of schedule.
                                     # But job_attr ensures we have data.
                                     # Better: check job existence in proj_jobs (structural).
                    # Note: Structural CPD uses all jobs in project.
                    pass
            
            # Rebuild pred_map purely from data for CPD (Structural)
            struct_pred = defaultdict(list)
            for j_id, job in jobs_map.items():
                if job["project_id"] == pid:
                    for succ in job["successors"]:
                        if succ in proj_jobs:
                            struct_pred[succ].append(j_id)
                            
            # Iterative Longest Path (Topological)
            # Topological sort logic via queue (Kahn's algorithm)
            in_degree = defaultdict(int)
            for u in proj_jobs:
                for v in proj_jobs:
                    if u in struct_pred[v]:
                        in_degree[v] += 1
                        
            # Queue for nodes with 0 in-degree (Sources)
            queue = []
            dist = {}
            # Initialize with source
            for u in proj_jobs:
                dist[u] = -float('inf')
                
            dist[src_id] = 0
            
            # Calculate In-degree for all nodes in this project
            temp_in_degree = {n: 0 for n in proj_jobs}
            for u in proj_jobs:
                for v in proj_jobs:
                    if u in struct_pred[v]:
                        temp_in_degree[v] += 1
            
            q_list = [n for n in proj_jobs if temp_in_degree[n] == 0]
            topo_order = []
            while q_list:
                u = q_list.pop(0)
                topo_order.append(u)
                # Update successors
                # We need reverse graph: succ -> preds
                # struct_pred[v] contains preds of v.
                # So successors of u are keys k such that u in struct_pred[k]
                for v in proj_jobs:
                    if u in struct_pred[v]:
                        temp_in_degree[v] -= 1
                        if temp_in_degree[v] == 0:
                            q_list.append(v)
                            
            # Process in topological order to find ES (Longest Path)
            for u in topo_order:
                # Update neighbors
                for v in proj_jobs:
                     if u in struct_pred[v]:
                         if dist[u] != -float('inf'):
                             new_dist = dist[u] + proj_jobs[u]
                             if new_dist > dist[v]:
                                 dist[v] = new_dist
            
            # CPD is ES(Sink)
            cpd = dist.get(sink_id, 0)
            project_cpds[pid] = {"cpd": cpd, "rel": rel_date, "sink": sink_id}
        
        # Calculate Total Project Delay (TPD)
        total_delay = 0.0
        for pid, info in project_cpds.items():
            sink_id = info["sink"]
            if sink_id in job_attr:
                sink_start = job_attr[sink_id]["start"]
                # Artificial jobs have 0 duration, so start = completion
                delay = sink_start - (info["rel"] + info["cpd"])
                total_delay += delay
            else:
                # If sink not scheduled, delay is huge? Or 0? 
                # Integrity check handles this, but default to 0 to prevent crash
                pass
            
        # Check if reported objective matches computed TPD
        if "objective_value" in solution:
            reported_obj = float(solution["objective_value"])
            diff = abs(reported_obj - total_delay)
            # Allow small numerical drift, but strict equality for feasibility
            if diff > 0.2:
                log_violation(7, f"Objective Value Mismatch (Reported vs Computed)", reported_obj, total_delay)

    except Exception as e:
        # Catch unexpected errors during execution
        log_violation(-1, f"Internal Execution Error: {str(e)}", 0.0, 1.0)

    # --- Format Output ---
    violations_list = []
    violated_indices = []
    magnitudes_list = []
    
    for idx in sorted(violations_dict.keys()):
        msg, lhs, rhs = violations_dict[idx]
        excess = lhs - rhs
        normalizer = rhs if rhs != 0 else 1.0
        ratio = excess / normalizer if normalizer != 0 else 0.0
        
        violations_list.append(msg)
        violated_indices.append(idx)
        magnitudes_list.append({
            "constraint": idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(excess),
            "normalizer": float(normalizer),
            "ratio": float(ratio)
        })

    return {
        "feasible": len(violations_dict) == 0,
        "violated_constraints": violated_indices,
        "violations": violations_list,
        "violation_magnitudes": magnitudes_list
    }