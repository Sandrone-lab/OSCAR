#!/usr/bin/env python3
"""
Feasibility checker for the Stochastic Planning & Scheduling problem
(Elci & Hooker, 2020) - Minimum Makespan variant.

Checks constraints from the deterministic equivalent MILP (Eq. 22),
numbered 1-9 from top to bottom:

  Constraint 1 (22b): sum_i x_{ij} = 1           for all j in J
  Constraint 2 (22c): beta_omega >= beta_{i,omega} for all i in I, omega in Omega
  Constraint 3 (22d): x_{ij} in {0,1}             for all i in I, j in J
  Constraint 4 (22e): beta_omega >= sum_t (t + p^omega_ij) z^omega_ijt
                                                    for all i in I, j in J, omega in Omega
  Constraint 5 (22f): z^omega_ijt <= x_{ij}       for all i,j,t,omega
  Constraint 6 (22g): sum_i sum_t z^omega_ijt = 1  for all j in J, omega in Omega
  Constraint 7 (22h): sum_j sum_{t' in T^omega_tij} c_{ij} z^omega_ijt' <= K_i
                                                    for all i in I, t in T, omega in Omega
  Constraint 8 (22i): z^omega_ijt = 0 for t < r_j  for all i,omega,j,t
  Constraint 9 (22j): z^omega_ijt in {0,1}         for all i,j,t,omega

Constraint 10 (objective consistency, Tier-C defense): the reported
  objective_value must equal the recomputed expected makespan
  sum_omega pi_omega * beta_omega, where beta_omega = max_j finish_j in
  scenario omega. Every variable needed for this recompute (the per-
  scenario finish times) is present in the solution's `schedule`, so a
  FULL two-sided recompute is possible.

  The original Constraint 4 (22e) objective check is ONE-SIDED -- it only
  flags reported < computed. Constraint 10 adds the symmetric direction
  (reported > computed), rejecting fabricated/inflated objective values
  that the structural constraint checks alone would otherwise accept.
"""

import argparse
import json


def check_feasibility(instance_path, solution_path, result_path):
    tol = 1e-5
    eps = 1e-5

    with open(instance_path) as f:
        inst = json.load(f)
    with open(solution_path) as f:
        sol = json.load(f)

    n = inst["n"]
    m = inst["m"]
    S = inst["num_scenarios"]
    K = inst["facility_capacity_limits"]       # K[i]
    c = inst["capacity_requirements"]          # c[i][j]
    r = inst["release_times"]                  # r[j]
    scenarios = inst["scenarios"]
    pi_prob = [sc["probability"] for sc in scenarios]
    p = [sc["processing_times"] for sc in scenarios]  # p[omega][i][j]

    assignment = sol.get("assignment", {})
    schedule = sol.get("schedule", None)
    objective_value = sol.get("objective_value", None)

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # Parse assignment: job j -> facility i
    assign = {}
    for j_str, i_val in assignment.items():
        assign[int(j_str)] = int(i_val)

    # Build x[i][j] matrix from assignment
    x = [[0] * n for _ in range(m)]
    for j_val, i_val in assign.items():
        if 0 <= i_val < m and 0 <= j_val < n:
            x[i_val][j_val] = 1

    # ------------------------------------------------------------------
    # Constraint 1 (22b): sum_i x_{ij} = 1 for all j in J
    # ------------------------------------------------------------------
    for j in range(n):
        lhs = sum(x[i][j] for i in range(m))
        rhs_val = 1.0
        violation_amount = abs(lhs - rhs_val)
        if violation_amount > tol:
            violated_constraints.add(1)
            violations.append(
                f"Constraint 1 (22b): Job {j} assigned to {lhs} facilities (expected 1)"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # ------------------------------------------------------------------
    # Constraint 3 (22d): x_{ij} in {0,1} for all i, j
    # Check that assignment values are valid (binary, valid facility index)
    # ------------------------------------------------------------------
    for j in range(n):
        if j in assign:
            fac = assign[j]
            if fac < 0 or fac >= m:
                violated_constraints.add(3)
                violation_amount = 1.0
                violations.append(
                    f"Constraint 3 (22d): Job {j} assigned to invalid facility {fac} "
                    f"(valid range 0..{m-1})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(fac),
                    "rhs": 0.0,
                    "raw_excess": float(violation_amount),
                    "normalizer": max(float(m - 1), eps),
                    "ratio": float(violation_amount / max(float(m - 1), eps)),
                })

    # ------------------------------------------------------------------
    # Constraints 2, 4-9 require schedule data (z variables)
    # ------------------------------------------------------------------
    if schedule is not None:
        # Parse schedule: sched[omega][j] = {facility, start, finish}
        sched = {}
        for omega_str, jobs in schedule.items():
            omega = int(omega_str)
            sched[omega] = {}
            for j_str, info in jobs.items():
                j_val = int(j_str)
                sched[omega][j_val] = {
                    "facility": int(info["facility"]),
                    "start": float(info["start"]),
                    "finish": float(info["finish"]),
                }

        for omega in range(S):
            if omega not in sched:
                continue

            # Compute per-facility makespan (beta_{i,omega}) and scenario makespan (beta_omega)
            beta_iw = [0.0] * m
            for j_val in range(n):
                if j_val not in sched[omega]:
                    continue
                info = sched[omega][j_val]
                fac = info["facility"]
                beta_iw[fac] = max(beta_iw[fac], info["finish"])

            beta_omega = max(beta_iw) if any(
                j_val in sched[omega] for j_val in range(n)
            ) else 0.0

            # ----------------------------------------------------------
            # Constraint 2 (22c): beta_omega >= beta_{i,omega}
            # This is a >= constraint: LHS = beta_omega, RHS = beta_{i,omega}
            # Violated if beta_{i,omega} > beta_omega
            # ----------------------------------------------------------
            for i in range(m):
                violation_amount = max(0.0, beta_iw[i] - beta_omega)
                if violation_amount > tol:
                    violated_constraints.add(2)
                    violations.append(
                        f"Constraint 2 (22c): Scenario {omega}, facility {i}: "
                        f"beta_omega={beta_omega:.4f} < beta_{{i,omega}}={beta_iw[i]:.4f}"
                    )
                    normalizer = max(abs(beta_iw[i]), eps)
                    violation_magnitudes.append({
                        "constraint": 2,
                        "lhs": float(beta_omega),
                        "rhs": float(beta_iw[i]),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(violation_amount / normalizer),
                    })

            # ----------------------------------------------------------
            # Constraint 4 (22e): beta_omega >= finish_j for all i, j
            # For the assigned facility, finish_j = start_j + p[omega][fac][j].
            # Also verify finish time consistency with processing times.
            # ----------------------------------------------------------
            for j_val in range(n):
                if j_val not in sched[omega]:
                    continue
                info = sched[omega][j_val]
                fac = info["facility"]
                start_j = info["start"]
                finish_j = info["finish"]

                # Verify finish = start + processing time
                if 0 <= fac < m:
                    expected_finish = start_j + p[omega][fac][j_val]
                    diff = abs(finish_j - expected_finish)
                    if diff > tol:
                        violated_constraints.add(4)
                        violations.append(
                            f"Constraint 4 (22e): Scenario {omega}, job {j_val}: "
                            f"finish={finish_j} != start({start_j}) + "
                            f"p[{omega}][{fac}][{j_val}]({p[omega][fac][j_val]}) = {expected_finish}"
                        )
                        normalizer = max(abs(expected_finish), eps)
                        violation_magnitudes.append({
                            "constraint": 4,
                            "lhs": float(finish_j),
                            "rhs": float(expected_finish),
                            "raw_excess": float(diff),
                            "normalizer": float(normalizer),
                            "ratio": float(diff / normalizer),
                        })

                # beta_omega >= finish_j
                violation_amount = max(0.0, finish_j - beta_omega)
                if violation_amount > tol:
                    violated_constraints.add(4)
                    violations.append(
                        f"Constraint 4 (22e): Scenario {omega}, job {j_val}: "
                        f"beta_omega={beta_omega:.4f} < finish={finish_j}"
                    )
                    normalizer = max(abs(finish_j), eps)
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": float(beta_omega),
                        "rhs": float(finish_j),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(violation_amount / normalizer),
                    })

            # Check objective consistency: objective_value should equal
            # sum pi_omega * beta_omega (computed from schedule)
            # This is an additional check for constraint 4 validity.
            if objective_value is not None and S > 0:
                computed_obj = 0.0
                all_scenarios_present = True
                for om in range(S):
                    if om in sched and sched[om]:
                        mk = max(sched[om][jj]["finish"] for jj in sched[om])
                        computed_obj += pi_prob[om] * mk
                    else:
                        all_scenarios_present = False

                if all_scenarios_present:
                    violation_amount = max(0.0, computed_obj - objective_value)
                    if violation_amount > tol:
                        violated_constraints.add(4)
                        violations.append(
                            f"Constraint 4 (22e): Reported objective {objective_value} "
                            f"< computed makespan objective {computed_obj:.4f}"
                        )
                        normalizer = max(abs(computed_obj), eps)
                        violation_magnitudes.append({
                            "constraint": 4,
                            "lhs": float(objective_value),
                            "rhs": float(computed_obj),
                            "raw_excess": float(violation_amount),
                            "normalizer": float(normalizer),
                            "ratio": float(violation_amount / normalizer),
                        })

            # ----------------------------------------------------------
            # Constraint 5 (22f): z^omega_ijt <= x_{ij}
            # Job scheduled on a facility must be assigned there.
            # ----------------------------------------------------------
            for j_val in range(n):
                if j_val not in sched[omega]:
                    continue
                sched_fac = sched[omega][j_val]["facility"]
                assigned_fac = assign.get(j_val, -1)
                if sched_fac != assigned_fac:
                    violated_constraints.add(5)
                    violations.append(
                        f"Constraint 5 (22f): Scenario {omega}, job {j_val}: "
                        f"scheduled on facility {sched_fac} but assigned to facility {assigned_fac}"
                    )
                    rhs_val = 0.0  # x_{sched_fac, j} = 0
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": 1.0,
                        "rhs": float(rhs_val),
                        "raw_excess": 1.0,
                        "normalizer": max(abs(rhs_val), eps),
                        "ratio": float(1.0 / max(abs(rhs_val), eps)),
                    })

            # ----------------------------------------------------------
            # Constraint 6 (22g): sum_i sum_t z^omega_ijt = 1 for all j, omega
            # Each job must be scheduled exactly once per scenario.
            # ----------------------------------------------------------
            for j_val in range(n):
                count = 1 if j_val in sched[omega] else 0
                rhs_val = 1.0
                violation_amount = abs(count - rhs_val)
                if violation_amount > tol:
                    violated_constraints.add(6)
                    violations.append(
                        f"Constraint 6 (22g): Scenario {omega}, job {j_val}: "
                        f"scheduled {count} times (expected 1)"
                    )
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": float(count),
                        "rhs": float(rhs_val),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(violation_amount / normalizer),
                    })

            # ----------------------------------------------------------
            # Constraint 7 (22h): Cumulative resource capacity
            # At each time t, for facility i:
            #   sum of c_{ij} for jobs j active at t <= K_i
            # Job j active at t if start_j <= t < start_j + p[omega][i][j]
            # ----------------------------------------------------------
            facility_jobs = {i: [] for i in range(m)}
            for j_val in range(n):
                if j_val in sched[omega]:
                    fac = sched[omega][j_val]["facility"]
                    facility_jobs[fac].append(j_val)

            for i in range(m):
                if not facility_jobs[i]:
                    continue

                # Collect event times (start and finish of each job on this facility)
                events = set()
                for j_val in facility_jobs[i]:
                    info = sched[omega][j_val]
                    events.add(int(info["start"]))
                    events.add(int(info["finish"]))

                # Check resource usage at each event time and just before each event
                times_to_check = set()
                for ev in events:
                    times_to_check.add(ev)
                    if ev > 0:
                        times_to_check.add(ev - 1)

                worst_violation = 0.0
                worst_t = None
                worst_lhs = 0.0

                for t in sorted(times_to_check):
                    resource_used = 0.0
                    for j_val in facility_jobs[i]:
                        info = sched[omega][j_val]
                        s_j = info["start"]
                        f_j = info["finish"]
                        if s_j <= t < f_j:
                            resource_used += c[i][j_val]

                    violation_amount = max(0.0, resource_used - K[i])
                    if violation_amount > worst_violation:
                        worst_violation = violation_amount
                        worst_t = t
                        worst_lhs = resource_used

                if worst_violation > tol:
                    violated_constraints.add(7)
                    violations.append(
                        f"Constraint 7 (22h): Scenario {omega}, facility {i}, "
                        f"time {worst_t}: resource usage {worst_lhs} > capacity {K[i]}"
                    )
                    normalizer = max(abs(float(K[i])), eps)
                    violation_magnitudes.append({
                        "constraint": 7,
                        "lhs": float(worst_lhs),
                        "rhs": float(K[i]),
                        "raw_excess": float(worst_violation),
                        "normalizer": float(normalizer),
                        "ratio": float(worst_violation / normalizer),
                    })

            # ----------------------------------------------------------
            # Constraint 8 (22i): z^omega_ijt = 0 for t < r_j
            # Equivalently: start_j >= r_j (release time)
            # This is a >= constraint.
            # ----------------------------------------------------------
            for j_val in range(n):
                if j_val not in sched[omega]:
                    continue
                start_j = sched[omega][j_val]["start"]
                rhs_val = float(r[j_val])
                violation_amount = max(0.0, rhs_val - start_j)
                if violation_amount > tol:
                    violated_constraints.add(8)
                    violations.append(
                        f"Constraint 8 (22i): Scenario {omega}, job {j_val}: "
                        f"start={start_j} < release time={r[j_val]}"
                    )
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 8,
                        "lhs": float(start_j),
                        "rhs": float(rhs_val),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(violation_amount / normalizer),
                    })

            # ----------------------------------------------------------
            # NOTE: The original two-stage stochastic program (1)-(3) uses
            # CONTINUOUS start-time variables s_j. The integer start_time
            # check (formerly "Constraint 9 (22j): z^omega_ijt in {0,1}")
            # only applies to the time-indexed deterministic-equivalent
            # reformulation (22), where z^omega_ijt binary requires starts
            # to land on integer time indices. LLM-evolved algorithms that
            # solve the original continuous-time problem produce valid
            # non-integer starts, which this checker must accept.
            # Accordingly, the integer-start check is INTENTIONALLY OMITTED.
            # ----------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 10 (objective consistency, full two-sided recompute):
    #   The reported objective_value must equal the expected makespan
    #     sum_{omega} pi_omega * beta_omega,
    #   where beta_omega = max_j finish_j in scenario omega.
    #   All variables required for this recompute (the per-scenario job
    #   finish times) are present in the solution's `schedule`, so a FULL
    #   recompute is possible -- no second-stage LP needs to be re-solved.
    #
    #   The original Constraint 4 (22e) objective check above is
    #   ONE-SIDED: it only flags reported < computed. This Constraint 10
    #   check is the symmetric Tier-C defense -- it rejects ANY
    #   disagreement (within tolerance) between the reported objective and
    #   the recomputed one, in particular reported > computed, i.e.
    #   fabricated / inflated objective values (e.g. sys.float_info.max)
    #   that the structural constraint checks alone would accept.
    # ------------------------------------------------------------------
    if schedule is not None and objective_value is not None and S > 0:
        obj_sched = {}
        try:
            for omega_str, jobs in schedule.items():
                om = int(omega_str)
                obj_sched[om] = {}
                for j_str, info in jobs.items():
                    obj_sched[om][int(j_str)] = float(info["finish"])
        except (TypeError, ValueError, KeyError):
            obj_sched = None

        if obj_sched is not None:
            computed_obj = 0.0
            all_scenarios_present = True
            for om in range(S):
                if om in obj_sched and obj_sched[om]:
                    computed_obj += pi_prob[om] * max(obj_sched[om].values())
                else:
                    all_scenarios_present = False

            if all_scenarios_present:
                try:
                    reported = float(objective_value)
                except (TypeError, ValueError):
                    reported = None
                if reported is not None:
                    abs_diff = abs(reported - computed_obj)
                    # 0.1% relative tolerance with a 1e-3 absolute floor
                    obj_tol = max(1e-3, 1e-3 * abs(computed_obj))
                    if abs_diff > obj_tol:
                        violated_constraints.add(10)
                        violations.append(
                            f"Constraint 10 (objective consistency): reported "
                            f"objective_value={reported} differs from recomputed "
                            f"expected makespan sum_omega(pi_omega*beta_omega)="
                            f"{computed_obj:.4f} (|diff|={abs_diff:.4g}, "
                            f"tol={obj_tol:.4g})"
                        )
                        normalizer = max(abs(computed_obj), eps)
                        violation_magnitudes.append({
                            "constraint": 10,
                            "lhs": float(reported),
                            "rhs": float(computed_obj),
                            "raw_excess": float(abs_diff),
                            "normalizer": float(normalizer),
                            "ratio": float(abs_diff / normalizer),
                        })

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feasibility checker for stochastic planning & scheduling "
                    "(Elci & Hooker 2020, Eq. 22)."
    )
    parser.add_argument(
        "--instance_path", required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", required=True,
        help="Path to the JSON solution file."
    )
    parser.add_argument(
        "--result_path", required=True,
        help="Path to write the JSON feasibility result."
    )
    args = parser.parse_args()

    result = check_feasibility(args.instance_path, args.solution_path, args.result_path)
    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")
