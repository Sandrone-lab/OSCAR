"""
Build the Simulator oracle once and keep it.

Section 2.1 of the paper: "Once built, the oracle is fixed, and every comparison
task runs through it rather than through an LLM." Building it fresh inside every
OSCAR run is therefore not what the framework describes, and it is expensive --
the draft alone is the slowest single call in the pipeline.

So it is built here, once, and every run loads the same certified artifact. The
build's cost is still charged to each run: the metrics records are saved next to
the code and replayed into each run's llm_metrics.jsonl, so an OSCAR run is
priced as if it had built its own simulator. No run gets a free oracle; we just
stop paying the wall clock over and over.

  python build_simulator.py                # build if absent
  python build_simulator.py --force        # rebuild from scratch
  python build_simulator.py --check        # re-certify the saved one, no LLM calls

Artifacts:
  simulator_certified.py            the oracle every run loads
  simulator_build.json              trials, revisions, wall clock, token cost
  simulator_build_metrics.jsonl     per-call records, replayed into each run
"""

import argparse
import json
import os
import shutil
import sys

from runlock import run_dir_lock, RunDirBusy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import problem

# every simulator artifact is problem-scoped: with OSCAR_PROBLEM unset these
# resolve to BASE_DIR, i.e. exactly the original MMRCPSP paths
SIM_PATH = os.path.join(problem.PROBLEM_DIR, "simulator_certified.py")
BUILD_LOG = os.path.join(problem.PROBLEM_DIR, "simulator_build.json")
BUILD_METRICS = os.path.join(problem.PROBLEM_DIR, "simulator_build_metrics.jsonl")
EXAMPLES_PATH = os.path.join(problem.PROBLEM_DIR, "certification_examples.json")

# Point a run at an oracle built outside the problem directory. The
# rich-message simulators live on a plain local drive, away from the Dropbox
# tree, and a run has to load one without the artifacts moving back.
#
# The build metrics travel with it. A run is priced as though it had built its
# own oracle, so charging it the OLD build's cost while running the NEW oracle
# would understate exactly the thing the rebuild exists to measure.
_ENV_SIM = os.environ.get("OSCAR_SIMULATOR_PATH", "").strip()
if _ENV_SIM:
    SIM_PATH = _ENV_SIM
    _sidecar = os.path.join(os.path.dirname(_ENV_SIM),
                            "simulator_build_metrics.jsonl")
    if os.path.exists(_sidecar):
        BUILD_METRICS = _sidecar

SIM_MODEL = os.environ.get("OSCAR_LARGE_MODEL", "qwen3.6-flash")   # 3.5-flash could not produce a certifiable oracle


PARTIAL_PATH = os.path.join(problem.PROBLEM_DIR, "partial_examples.json")


def load_examples():
    """Recorded examples, plus per-rule fragments when the problem supplies them.

    A fragment is an optional piece of the additional information shipped with a
    problem: a small excerpt of a solution, labelled acceptable or not, showing
    what one individual rule allows. Fragments are used only while the simulator
    is being built; nothing in an OSCAR run ever sends one.
    """
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if os.path.exists(PARTIAL_PATH):
        with open(PARTIAL_PATH, "r", encoding="utf-8") as f:
            for ex in json.load(f):
                ex["partial"] = True
                examples.append(ex)
    return examples

def _ensure_api_key():
    """Ask for the key if the shell has not set one, the way start.py does."""
    import getpass
    if os.environ.get("OSCAR_API_KEY"):
        return True
    try:
        k = getpass.getpass("OSCAR_API_KEY (typing hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if not k:
        return False
    os.environ["OSCAR_API_KEY"] = k
    return True

def _fill_env_from_settings():
    """Take the endpoint and model names from settings.txt when the shell has
    not set them, by calling run_experiment.load_settings() itself -- the same
    parser start.py uses, so the two cannot drift apart. The key is never in
    that file; it still has to come from the environment.
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "run_experiment.py")
    if not os.path.exists(path):
        return
    try:
        spec = importlib.util.spec_from_file_location("oscar_settings", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg, _ = mod.load_settings()
    except Exception:
        return
    for var, key in (("OSCAR_API_BASE", "api_base"),
                     ("OSCAR_SMALL_MODEL", "small_model"),
                     ("OSCAR_LARGE_MODEL", "large_model")):
        if not os.environ.get(var) and cfg.get(key):
            os.environ[var] = cfg[key]


def load_certified_simulator():
    """Return the feasibility_check function every OSCAR run shares."""
    from generate_simulator import _load_module
    if not os.path.exists(SIM_PATH):
        raise RuntimeError(
            "No certified simulator at %s. Build it once with:\n"
            "    python build_simulator.py" % SIM_PATH)
    return _load_module(SIM_PATH).feasibility_check


def build_cost_records():
    """The build's per-call metrics, to be charged to a run."""
    if not os.path.exists(BUILD_METRICS):
        return []
    out = []
    with open(BUILD_METRICS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["amortized"] = True      # spent once, charged to every run
                out.append(rec)
    return out


def charge_build_to(metrics_path):
    """Append the shared build's cost to one run's metrics."""
    records = build_cost_records()
    if not records:
        return 0
    with open(metrics_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def check_saved(verbose=True):
    from generate_simulator import _check_examples
    fn = load_certified_simulator()
    ok, failed, review, _exc = _check_examples(fn, load_examples(), problem.PROBLEM_DIR,
                                               verbose=verbose)
    return ok, failed, review


def main():
    if not _ensure_api_key():
        sys.stderr.write("No API key; nothing can be called.\n")
        return 1
    ap = argparse.ArgumentParser()
    _fill_env_from_settings()
    ap.add_argument("--force", action="store_true", help="rebuild even if one exists")
    ap.add_argument("--check", action="store_true",
                    help="re-certify the saved simulator without calling any LLM")
    ap.add_argument("--model", default=SIM_MODEL)
    ap.add_argument("--max-trials", type=int, default=8)
    # How many identical failures before the build is called stuck. The
    # default suits a short build; a long one needs more patience or it
    # stops long before its trial budget is spent.
    ap.add_argument("--stuck-after", type=int, default=3)
    # Build to a different filename so an experimental oracle cannot overwrite the
    # one Section 5.1's results were produced with. Every 5.1 number is tied to a
    # specific simulator; replacing it in place would silently invalidate them.
    ap.add_argument("--out", default=None,
                    help="write the certified simulator here instead of "
                         "simulator_certified.py (filename or absolute path)")
    args = ap.parse_args()
    if args.out:
        global SIM_PATH
        SIM_PATH = (args.out if os.path.isabs(args.out)
                    else os.path.join(problem.PROBLEM_DIR, args.out))

    examples = load_examples()
    print("%d certification example(s): %d feasible, %d infeasible"
          % (len(examples), sum(1 for e in examples if e["feasible"]),
             sum(1 for e in examples if not e["feasible"])),
          "  of which %d are per-rule fragments" % sum(1 for e in examples if e.get("partial")))

    if args.check:
        ok, failed, _ = check_saved()
        print("\nsaved simulator certified:", ok)
        return 0 if ok else 1

    if os.path.exists(SIM_PATH) and not args.force:
        print("\n%s already exists; re-certifying it (no LLM calls)." % os.path.basename(SIM_PATH))
        ok, failed, _ = check_saved()
        if ok:
            print("\nAlready certified. Use --force to rebuild.")
            return 0
        print("\nSaved simulator no longer certifies; rebuilding.")

    from generate_simulator import build_certified_simulator
    import utils

    # Two builds writing the same metrics file corrupts the per-run cost, and
    # the loser silently overwrites the winner's candidate file.
    try:
        lock = run_dir_lock(problem.PROBLEM_DIR)
        lock.__enter__()
    except RunDirBusy as e:
        print("Another simulator build is already running: %s" % e)
        return 1

    # isolate this build's metrics so its cost can be charged to each run later
    os.environ["LLM_METRICS_PATH"] = BUILD_METRICS
    open(BUILD_METRICS, "w").close()

    print("\nBuilding (this is the slowest step in the pipeline; progress below)\n")
    tmp_path = os.path.join(problem.PROBLEM_DIR, "simulator_candidate.py")
    mod, log = build_certified_simulator(
        open(problem.DESC_PATH, encoding="utf-8").read(),
        examples, args.model, tmp_path, base_dir=problem.PROBLEM_DIR,
        max_trials=args.max_trials, verbose=True,
        stuck_after=args.stuck_after)

    summary = utils.summarize_metrics(BUILD_METRICS)
    log["metrics"] = summary
    log["model"] = args.model
    with open(BUILD_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    if not log.get("certified"):
        print("\nNOT CERTIFIED. Nothing was saved; the candidate is at %s for inspection."
              % os.path.basename(tmp_path))
        print(json.dumps(log["failures"][-2:], indent=2, default=str))
        return 1

    shutil.move(tmp_path, SIM_PATH)
    print("\nSaved %s" % os.path.basename(SIM_PATH))
    print("  trials=%d syntax_fixes=%d revisions=%d  %.0fs"
          % (log["trials"], log["syntax_fixes"], log["revisions"], log["seconds"]))
    print("  %d LLM call(s), %d tokens -- charged to every OSCAR run"
          % (summary["calls"], summary["total_tokens"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
