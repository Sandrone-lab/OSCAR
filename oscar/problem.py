"""
Problem interface for the harness -- now loadable per problem.

Set OSCAR_PROBLEM to a problem directory (e.g. "new questions/QWEN3") and every
part of the harness -- drivers, grading, simulator builder, summariser -- works
against that problem: its desc.txt, its trusted feasibility_check.py, its
instances and reference optima. Unset, it defaults to the original MMRCPSP
problem in this directory, byte-for-byte as before.

A problem directory must contain:
    desc.txt                 the problem description given to every model
    feasibility_check.py     the trusted checker (ground truth for scoring)
    instance/                tiny_instance.json, large_instance_N.json
    gurobi_solution/         reference solutions with objective_value
Optional:
    problem_config.json      overrides: name, schema fields, objective label

Objective recomputation: the FrontierOR checkers all verify objective
consistency as part of feasibility, so when the checker passes, the reported
objective is trustworthy and is used for scoring. A problem can still supply an
independent recomputation via problem_config.json ("recompute_module").
"""

import importlib.util
import threading
import json
import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROBLEM_DIR = os.environ.get("OSCAR_PROBLEM")

_cfg = {}
_cfg_path = os.path.join(PROBLEM_DIR, "problem_config.json")
if os.path.exists(_cfg_path):
    _cfg = json.load(open(_cfg_path, encoding="utf-8"))

NAME = _cfg.get("name") or os.path.basename(PROBLEM_DIR.rstrip("\\/"))
DESC_PATH = os.path.join(PROBLEM_DIR, "desc.txt")
OBJECTIVE_NAME = _cfg.get("objective_name", "obj")
OBJECTIVE_SENSE = _cfg.get("objective_sense", "min")

# Solution schema, used only to tell a malformed file from a bad answer.
# Heterogeneous problems may define no list field at all; grading degrades to
# "must be a JSON object carrying objective_value".
SOLUTION_LIST_FIELD = _cfg.get("solution_list_field")
SOLUTION_ENTRY_FIELDS = tuple(_cfg.get("solution_entry_fields", ()))
SOLUTION_SCALAR_FIELDS = tuple(_cfg.get("solution_scalar_fields", ("objective_value",)))

# Top-level fields a solution is expected to carry. Taken from the problem's own
# solution schema when it ships one, so the robustness battery can mutate every
# declared field even for problems whose decisions are not a single list of
# uniform entries (a graph colouring keyed by vertex, say).
SOLUTION_FIELDS = ()
_schema_path = os.path.join(PROBLEM_DIR, "solution_schema.json")
if os.path.exists(_schema_path):
    try:
        with open(_schema_path, "r", encoding="utf-8") as _f:
            _schema = json.load(_f)
        SOLUTION_FIELDS = tuple((_schema.get("properties") or _schema).keys())
    except Exception:
        SOLUTION_FIELDS = ()

if PROBLEM_DIR == BASE_DIR and not _cfg:
    # original MMRCPSP defaults, unchanged
    NAME = "MMRCPSP (FrontierOR araujo2020)"
    OBJECTIVE_NAME = "TPD"
    SOLUTION_LIST_FIELD = "schedule"
    SOLUTION_ENTRY_FIELDS = ("job_id", "mode_id", "start_time")
    SOLUTION_SCALAR_FIELDS = ("makespan",)


# --- trusted checker ---------------------------------------------------------

_checker_mod = None
# Two threads importing the checker at once leave one of them holding a module
# that is still executing, and its half-initialised globals then fabricate
# violations against perfectly good solutions.  This surfaced once as a graded
# arm scoring 4/5 when the true answer was 5/5, so the import is serialised and
# the whole call is held under the same lock: checker modules are third-party
# code and are not guaranteed re-entrant.
_CHECK_LOCK = threading.RLock()


def _load_checker():
    global _checker_mod
    with _CHECK_LOCK:
        if _checker_mod is None:
            path = os.path.join(PROBLEM_DIR, "feasibility_check.py")
            spec = importlib.util.spec_from_file_location("problem_checker", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _checker_mod = mod          # publish only once fully executed
        return _checker_mod


def check_feasibility(instance, solution):
    """The trusted checker. Ground truth for scoring only.

    Handles both FrontierOR signatures: (instance, solution) dicts, and the
    path-based (instance_path, solution_path, result_path) variant.
    """
    mod = _load_checker()
    fn = mod.check_feasibility
    try:
        code = fn.__code__
        params = code.co_varnames[:code.co_argcount]
    except AttributeError:
        params = ()
    if params[:2] == ("instance", "solution") or len(params) == 2:
        out = fn(instance, solution)
        if isinstance(out, tuple):
            # (feasible, violated_constraints, violations, magnitudes) style
            parts = list(out) + [None] * 4
            return {"feasible": bool(parts[0]),
                    "violated_constraints": sorted(parts[1] or []),
                    "violations": list(parts[2] or []),
                    "violation_magnitudes": list(parts[3] or [])}
        return out
    # path-based checker: round-trip through temp files
    with tempfile.TemporaryDirectory() as td:
        ip, sp, rp = (os.path.join(td, n) for n in ("i.json", "s.json", "r.json"))
        json.dump(instance, open(ip, "w"))
        json.dump(solution, open(sp, "w"))
        fn(ip, sp, rp)
        return json.load(open(rp, encoding="utf-8"))


def recompute_objective(instance, solution):
    """Objective for scoring.

    If the problem supplies an independent recomputation module, use it. For
    the original MMRCPSP problem the recomputation lives here. Otherwise fall
    back to the reported objective, but ONLY when the trusted checker (which
    verifies objective consistency on every FrontierOR problem) accepts the
    solution -- an unverified self-report is never used.
    """
    mod_name = _cfg.get("recompute_module")
    if mod_name:
        path = os.path.join(PROBLEM_DIR, mod_name)
        spec = importlib.util.spec_from_file_location("problem_recompute", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.recompute_objective(instance, solution)
    try:
        if check_feasibility(instance, solution).get("feasible"):
            v = solution.get("objective_value")
            return float(v) if v is not None else None
        return None
    except Exception:
        return None


# --- instances and reference optima ------------------------------------------

def instance_path(name):
    """'tiny' or 'large_N' -> path. Falls back to a literal filename."""
    idir = os.path.join(PROBLEM_DIR, "instance")
    if os.path.isdir(idir):
        if name == "tiny":
            return os.path.join(idir, "tiny_instance.json")
        if name.startswith("large_"):
            return os.path.join(idir, "large_instance_%s.json" % name.split("_", 1)[1])
        # any other short name naming an <name>_instance.json in the instance dir
        cand = os.path.join(idir, "%s_instance.json" % name)
        if os.path.exists(cand):
            return cand
        if os.path.exists(os.path.join(idir, name)):
            return os.path.join(idir, name)
    return os.path.join(PROBLEM_DIR, name)


def known_optimum(instance_name):
    """Objective of the Gurobi reference solution for that instance, if optimal."""
    base = os.path.basename(str(instance_name))
    if PROBLEM_DIR == BASE_DIR:
        legacy = {"data.json": 3.0, "large_instance_1.json": 0.0, "large_instance_2.json": 1.0,
                  "large_instance_3.json": 0.0, "large_instance_4.json": 4.0,
                  "large_instance_5.json": 0.0}
        return legacy.get(base)
    if base == "tiny_instance.json" or instance_name == "tiny":
        sol = "tiny_solution.json"
    elif "large" in base:
        n = "".join(c for c in base if c.isdigit()) or "1"
        sol = "large_solution_%s.json" % n
    elif base.endswith("_instance.json"):
        sol = base.replace("_instance.json", "_solution.json")
    else:
        return None
    p = os.path.join(PROBLEM_DIR, "gurobi_solution", sol)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
        status = str(d.get("status_str") or d.get("status_name") or d.get("status") or "").lower()
        if status and status not in ("2", "optimal"):
            return None                     # reference not proven optimal
        if _cfg.get("restate_known_optimum") and _cfg.get("recompute_module"):
            # Score everything, the reference included, under the problem's own
            # recomputation rule, so candidate and optimum share one objective.
            inst = json.load(open(str(instance_path(instance_name)), encoding="utf-8"))
            return recompute_objective(inst, d)
        return d.get("objective_value")
    except Exception:
        return None
