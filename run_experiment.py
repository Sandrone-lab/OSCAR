# -*- coding: utf-8 -*-
"""One experiment from the OSCAR paper: one problem, one method, n repetitions.

The same runs start.py makes, as a single non-interactive command -- for a loop over
several problems, a batch left overnight, or a cluster job. Nothing here is harder than
the menu; it just does not ask.

What to run comes from settings.txt beside this file -- the two models, the endpoint,
the problem, how many repetitions. A flag here overrides it; an environment variable
overrides settings.txt but loses to a flag.
echo %OSCAR_API_BASE%
UNIFORM below is the other layer, and it is not meant to be edited: five values held
identical for all five problems -- no solver time limit, the same memory ceiling --
so that a number from one problem can be compared with a number from another.

    export OSCAR_API_KEY=...        # your model API key. Never put it in settings.txt
    export COPT_LICENSE_DIR=...     # a COPT licence

    python run_experiment.py --method oscar               # what settings.txt says
    python run_experiment.py --method oneshot --reps 10   # ...but ten repetitions
    python run_experiment.py --method oscar --problem elci2022 --n-per-level 1,1,3,3

Expect to wait. Most of a run is the endpoint answering and COPT solving each program
it is given; see "How long a run takes" in README.md.
"""
import argparse
import glob
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OSCAR = os.path.join(HERE, "oscar")
PROBLEMS = os.path.join(HERE, "problems")
SETTINGS_PATH = os.path.join(HERE, "settings.txt")

UNIFORM = {
    "OSCAR_SOLVER_TL": "0",              # the Coder is not told to set a solver time limit
    "OSCAR_SOLVER_TIME_CAP": "0",        # and none is imposed afterwards
    "OSCAR_SOLVER_MEMLIMIT_MB": "3072",      # COPT MemLimit, same for every problem
    "OSCAR_PROCESS_MEMLIMIT_MB": "3456",    # in-process guard, MemLimit + 384
    "OSCAR_HASMIPSOL_SHIM": "1",         # coptpy exposes hasmipsol as an int in some builds
}

# The settings a reader edits, and their built-in values. A key that is not here is
# reported rather than ignored, so a typo says so instead of quietly doing nothing.
DEFAULT_SETTINGS = {
    "small_model": "qwen3.5-flash",
    "large_model": "qwen3.6-flash",
    "api_base": "",
    "problem": "mehrotra1996",
    "oneshot_reps": "5",
    "oneshot_model": "small",
    "oscar_runs": "1",
    "oscar_n_per_level": "1",
}


def load_settings(path=SETTINGS_PATH):
    """Read settings.txt. Returns (settings, complaints); the file is never written."""
    s = dict(DEFAULT_SETTINGS)
    bad = []
    try:
        lines = io.open(path, encoding="utf-8").read().splitlines()
    except IOError:
        return s, ["%s not found; using the built-in defaults" % os.path.basename(path)]
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(" #", 1)[0].strip()      # a trailing comment, not a URL fragment
        if "=" not in line:
            bad.append("line %d: expected  key = value" % i)
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k not in DEFAULT_SETTINGS:
            bad.append("line %d: unknown setting %r" % (i, k))
            continue
        s[k] = v

    for k in ("small_model", "large_model"):
        if not s[k]:
            bad.append("%s is empty; using %s" % (k, DEFAULT_SETTINGS[k]))
            s[k] = DEFAULT_SETTINGS[k]
    if s["small_model"] == s["large_model"]:
        bad.append("small_model and large_model are both %r. OSCAR uses two models and "
                   "exactly two: with one named twice the Escalator has nothing to "
                   "escalate to." % s["small_model"])
    if s["oneshot_model"] not in ("small", "large"):
        bad.append("oneshot_model must be small or large, not %r" % s["oneshot_model"])
        s["oneshot_model"] = DEFAULT_SETTINGS["oneshot_model"]
    if s["problem"] not in os.listdir(PROBLEMS):
        bad.append("problem %r is not a directory under problems/; using %s"
                   % (s["problem"], DEFAULT_SETTINGS["problem"]))
        s["problem"] = DEFAULT_SETTINGS["problem"]
    for k in ("oneshot_reps", "oscar_runs"):
        if not s[k].isdigit() or int(s[k]) < 1:
            bad.append("%s must be a whole number of 1 or more, not %r" % (k, s[k]))
            s[k] = DEFAULT_SETTINGS[k]
    parts = [x.strip() for x in s["oscar_n_per_level"].split(",")]
    if not all(p.isdigit() for p in parts):
        bad.append("oscar_n_per_level must be a number, or numbers separated by commas, "
                   "not %r" % s["oscar_n_per_level"])
        s["oscar_n_per_level"] = DEFAULT_SETTINGS["oscar_n_per_level"]
    else:
        s["oscar_n_per_level"] = ",".join(parts)
    return s, bad


def main():
    cfg, complaints = load_settings()
    for c in complaints:
        sys.stderr.write("settings.txt: %s\n" % c)
    coder = cfg["small_model"] if cfg["oneshot_model"] == "small" else cfg["large_model"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=cfg["problem"],
                    choices=sorted(os.listdir(PROBLEMS)))
    ap.add_argument("--method", required=True, choices=["oscar", "oneshot"])
    ap.add_argument("--model", default=coder, help="one-shot only")
    ap.add_argument("--reps", type=int, default=int(cfg["oneshot_reps"]),
                    help="one-shot repetitions")
    ap.add_argument("--runs", type=int, default=int(cfg["oscar_runs"]), help="OSCAR runs")
    ap.add_argument("--n-per-level", default=cfg["oscar_n_per_level"],
                    help='attempts at each menu configuration: "1", or "1,1,3,3" for '
                         'one per level in cost order')
    ap.add_argument("--proc-tl", type=int, default=1800,
                    help="wall-clock bound on one generated program, seconds")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    env = dict(os.environ)
    env.update(UNIFORM)
    env["OSCAR_PROBLEM"] = os.path.join(PROBLEMS, a.problem)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # settings.txt is the weakest source: a variable already in the environment wins.
    for var, key in (("OSCAR_SMALL_MODEL", "small_model"),
                     ("OSCAR_LARGE_MODEL", "large_model"),
                     ("OSCAR_API_BASE", "api_base")):
        if not env.get(var) and cfg[key]:
            env[var] = cfg[key]
    for var, why in (("OSCAR_API_KEY", "no model can be called"),
                     ("COPT_LICENSE_DIR", "COPT runs size-limited and every result is wrong")):
        if not env.get(var):
            sys.stderr.write("%s is not set: %s\n" % (var, why))

    out = os.path.join(a.outdir, a.problem, a.method)
    os.makedirs(out, exist_ok=True)
    # Both tools number their output directories from an index, so a second invocation
    # would land on run_00 again and overwrite it. Start after whatever is already on
    # disk instead: asking for 5 twice gives you 10, and a job you interrupted resumes.
    if a.method == "oscar":
        done = len(glob.glob(os.path.join(out, "tiny_instance", "run_*")))
        if done:
            print("%d run(s) already here; continuing from run_%02d" % (done, done))
        cmd = [sys.executable, "-u", "oscar_escalator.py",
               "--instance", "tiny_instance.json",
               "--start-run", str(done), "--runs", str(done + a.runs),
               "--n-per-level", str(a.n_per_level),
               "--proc-tl", str(a.proc_tl), "--outdir", out]
    else:
        done = len(glob.glob(os.path.join(out, a.model, "tiny_instance", "rep_*")))
        if done:
            print("%d repetition(s) already here; continuing from rep_%02d" % (done, done))
        cmd = [sys.executable, "-u", "oneshot.py",
               "--model", a.model, "--instance", "tiny_instance.json",
               "--start-rep", str(done), "--reps", str(done + a.reps),
               "--max-debug", "10",
               "--proc-tl", str(a.proc_tl), "--outdir", out]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=OSCAR, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
