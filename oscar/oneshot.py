"""
One-shot baseline for Section 5.1: OSCAR versus one-shot generation.

One repetition is exactly the first Coder call of an OSCAR session, plus the
syntax-repair loop that existing one-shot pipelines are allowed ("pipelines
repair code that fails to run, while the correctness of the formulation itself
goes unchecked"). There is no Simulator, no Reviewer, and no second
formulation attempt: whatever the first runnable program produces is the answer.

Prompt parity with OSCAR is guaranteed by importing the prompt builders from
our_model.py rather than restating them here.

Usage:
  python oneshot.py --model qwen3.5-flash --instance data.json --reps 10
  python oneshot.py --model qwen3.6-flash --instance large_instance_2.json --reps 10

Results land in run_oneshot/<model>/<instance stem>/rep_XX/ with one
summary.jsonl row per repetition.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_CODE = os.path.join(BASE_DIR, "run_code.py")

from grading import safe_check_feasibility, recompute_objective
import problem
from copt_env import ensure_license, looks_unlicensed, solver_slot
from utils import summarize_metrics, call_role
from runlock import run_dir_lock, RunDirBusy

# One-shot's own repair rule, set by the user on 2026-09-11: syntax repair is always
# done by the SMALL model, whichever model wrote the program, for at most REPAIR_ROUNDS
# rounds, and a program still broken after that is a failed repetition. This keeps each
# column a measurement of the model in its header. Under the escalator's ladder it was
# not: 48% of all repair rounds landed on the large model, and three quarters of the
# "Qwen3.5" column's cost was Qwen3.6, with nearly every success coming from a
# repetition that had escalated.
REPAIR_MODEL = os.environ.get("OSCAR_SMALL_MODEL", "qwen3.5-flash")
REPAIR_ROUNDS = 10


STRUCT_MODEL = os.environ.get("OSCAR_SMALL_MODEL", "qwen3.5-flash")   # same structure constructor OSCAR uses
OMIT_STRUCTURE = "__OMIT__"      # must match our_model.OMIT_STRUCTURE


def _load_prompt_builders():
    """Import the Coder prompts from our_model so both arms share them verbatim."""
    import our_model
    return our_model.generate_optimization_code, our_model.revise_code


def run_once(model, instance_path, rep_dir, structure, desc, max_debug, proc_tl):
    """One independent one-shot attempt. Returns the result record.

    ``structure`` may be None, in which case the structure constructor is called
    here and charged to this repetition, matching OSCAR's preparations.
    """
    with run_dir_lock(rep_dir):
        return _run_once(model, instance_path, rep_dir, structure, desc,
                         max_debug, proc_tl)


def _log_repair_error(attempt_dir, attempt_no, error_message, code):
    """Record one failed run so recurring COPT errors can be found later."""
    try:
        with open(os.path.join(attempt_dir, "repair_log.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps({
                "attempt": attempt_no,
                "error": error_message[:4000],
                "first_line": next((ln for ln in reversed(error_message.strip().splitlines())
                                    if ln.strip()), "")[:300],
            }, ensure_ascii=False) + "\n")
        with open(os.path.join(attempt_dir, "code_attempt_%02d.py" % attempt_no),
                  "w", encoding="utf-8") as f:
            f.write(code)
    except Exception:
        pass          # logging must never break a run

def _run_once(model, instance_path, rep_dir, structure, desc, max_debug, proc_tl):
    metrics_path = os.path.join(rep_dir, "llm_metrics.jsonl")
    os.environ["LLM_METRICS_PATH"] = metrics_path

    generate_optimization_code, revise_code = _load_prompt_builders()
    if structure == OMIT_STRUCTURE:
        pass                      # ablation: the Coder never sees a structure
    elif structure is None:
        import our_model
        with call_role("prep_structure"):
            structure = our_model.generate_structure(desc, model=STRUCT_MODEL)
        with open(os.path.join(rep_dir, "structure.json"), "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2)

    with open(instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)
    data_keys = list(instance.keys())
    shutil.copy(instance_path, os.path.join(rep_dir, "data.json"))

    code_path = os.path.join(rep_dir, "code.py")
    solution_path = os.path.join(rep_dir, "solution.json")

    rec = {
        "model": model,
        "instance": os.path.basename(instance_path),
        "jobs": os.environ.get("OSCAR_JOBS"),
        "coder_calls": 0,
        "debug_calls": 0,
        "ran_ok": False,
        "timed_out": False,
        "solution_present": False,
        "gt_feasible": None,
        "gt_violated": None,
        "reported_objective": None,
        "recomputed_objective": None,
        "scalars": None,
        "schema_error": None,
        "gt_violations": None,
        "error": None,
    }

    started = time.perf_counter()

    # ---- the one shot -------------------------------------------------------
    try:
        with call_role("oneshot_coder"):
            code = generate_optimization_code(desc, structure, data_keys, model, "")
        rec["coder_calls"] = 1
    except Exception as e:
        rec["error"] = "coder call failed: %s" % e
        rec["wall_sec"] = round(time.perf_counter() - started, 3)
        return rec
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    # ---- syntax repair only: the formulation is never revisited -------------
    # Every round is run by the small model, up to max_debug rounds; a program still
    # broken at the end is a failed repetition. No escalation and no stall rule: the
    # point of a one-shot column is what ONE model does with its own mistakes.
    result = None
    for attempt in range(max_debug + 1):
        if os.path.exists(solution_path):
            os.remove(solution_path)
        try:
            with solver_slot():
                result = subprocess.run(
                    [sys.executable, RUN_CODE, "code.py"], cwd=rep_dir,
                    capture_output=True, text=True, timeout=proc_tl,
                )
        except subprocess.TimeoutExpired:
            rec["timed_out"] = True
            break
        if looks_unlicensed(result.stdout):
            rec["copt_unlicensed"] = True
        produced = os.path.exists(solution_path) and os.path.getsize(solution_path) > 0
        if result.returncode == 0 and produced:
            rec["ran_ok"] = True
            break
        if attempt == max_debug:
            rec["ran_ok"] = bool(result.returncode == 0 and produced)
            rec["repair_exhausted"] = not rec["ran_ok"]
            break
        parts = []
        if result.stderr:
            parts.append("STDERR:\n" + result.stderr)
        if result.stdout:
            parts.append("STDOUT:\n" + result.stdout)
        error_message = "\n".join(parts) or ("Return code: %d" % result.returncode)
        if result.returncode == 0 and not produced:
            rec["empty_solution_repairs"] = rec.get("empty_solution_repairs", 0) + 1
            error_message = ("The program exited normally but wrote no solution.json, or wrote an empty one. Do not catch the exception and write an empty file: fix the underlying error so a complete solution is written.\n\n"
                             + error_message)
        _log_repair_error(rep_dir, attempt, error_message, code)
        rec.setdefault("repair_models", []).append(REPAIR_MODEL)
        try:
            with call_role("oneshot_repair"):
                code = revise_code(desc, structure, data_keys, model=REPAIR_MODEL,
                                   review_context=error_message, previous_code=code)
            rec["debug_calls"] += 1
        except Exception as e:
            rec["error"] = "repair call failed: %s" % e
            break
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

    rec["wall_sec"] = round(time.perf_counter() - started, 3)

    # ---- grade --------------------------------------------------------------
    if rec["ran_ok"] and os.path.exists(solution_path) and os.path.getsize(solution_path) > 0:
        try:
            with open(solution_path, "r", encoding="utf-8") as f:
                solution = json.load(f)
            rec["solution_present"] = True
            rec["reported_objective"] = solution.get("objective_value")
            rec["scalars"] = {f: solution.get(f) for f in problem.SOLUTION_SCALAR_FIELDS}
            gt = safe_check_feasibility(instance, solution)
            rec["gt_feasible"] = gt["feasible"]
            rec["gt_violated"] = gt["violated_constraints"]
            rec["schema_error"] = gt.get("schema_error", False)
            rec["gt_violations"] = gt["violations"][:3]
            rec["recomputed_objective"] = recompute_objective(instance, solution)
        except Exception:
            rec["error"] = "grading failed: %s" % traceback.format_exc(limit=2)

    rec.update(summarize_metrics(metrics_path))
    return rec


def main():
    ensure_license()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. qwen3.5-flash or qwen3.6-flash")
    ap.add_argument("--instance", required=True, help="instance json in this directory")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-debug", type=int, default=REPAIR_ROUNDS,
                    help="syntax-repair calls allowed, matching OSCAR's inner loop")
    ap.add_argument("--proc-tl", type=int, default=int(os.environ.get("OSCAR_PROC_TL", 600)))
    _default_out = (os.path.join(BASE_DIR, "run_oneshot") if problem.PROBLEM_DIR == BASE_DIR
                    else os.path.join(BASE_DIR, "results", problem.NAME, "run_oneshot"))
    ap.add_argument("--outdir", default=_default_out)
    ap.add_argument("--start-rep", type=int, default=0, help="resume from this repetition")
    ap.add_argument("--structure", choices=["onthefly", "file", "none"], default="onthefly",
                    help="onthefly: build the structure per repetition, as OSCAR does; "
                         "file: reuse the checked-in structure.json")
    args = ap.parse_args()

    instance_path = (problem.instance_path(args.instance)
                     if not os.path.exists(os.path.join(BASE_DIR, args.instance))
                     else os.path.join(BASE_DIR, args.instance))
    stem = os.path.splitext(os.path.basename(args.instance))[0]
    out_root = os.path.join(args.outdir, args.model, stem)
    os.makedirs(out_root, exist_ok=True)

    with open(problem.DESC_PATH, "r", encoding="utf-8") as f:
        desc = f.read()
    structure = None
    if args.structure == "none":
        # Ablation: skip the structure constructor entirely, so the Coder gets
        # the description and the data keys and must infer the parameters and
        # their shapes itself.
        structure = OMIT_STRUCTURE
    elif args.structure == "file":
        with open(os.path.join(BASE_DIR, "structure.json"), "r", encoding="utf-8") as f:
            structure = f.read()

    summary_path = os.path.join(out_root, "summary.jsonl")
    for rep in range(args.start_rep, args.reps):
        rep_dir = os.path.join(out_root, "rep_%02d" % rep)
        print("=" * 60)
        print("%s | %s | rep %d/%d" % (args.model, args.instance, rep + 1, args.reps))
        rec = run_once(args.model, instance_path, rep_dir, structure, desc,
                       args.max_debug, args.proc_tl)
        rec["rep"] = rep
        with open(os.path.join(rep_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("  ran_ok=%s debug=%d gt_feasible=%s obj=%s tokens=%s" % (
            rec["ran_ok"], rec["debug_calls"], rec["gt_feasible"],
            rec["recomputed_objective"], rec.get("total_tokens")))

    print("\nSummary written to %s" % summary_path)


if __name__ == "__main__":
    main()
