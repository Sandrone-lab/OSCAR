# -*- coding: utf-8 -*-
"""OSCAR: a guided runner.

Everything here is one command of run_experiment.py, which runs the same experiments
without asking anything -- reach for that when you want a loop over several problems or
a batch left running overnight. This exists so nobody has to learn the flags first.
Changing the method rather than running it is a third thing again: that lives in
oscar/, and README.md says where to start.

What a run does is set in settings.txt beside this file: the two model names, the
endpoint, which problem, and how many repetitions or runs. Edit it in any text editor;
menu option 2 reloads it without restarting, and every prompt here offers what the file
says, so pressing Enter runs exactly what is written there.

On credentials. Your API key is never written to disk by this tool, never printed, and
never placed on a command line where it would show up in a process list. It is held in
memory for as long as this program runs and passed to each experiment through the
environment. You can instead set OSCAR_API_KEY yourself before starting, and this tool
will use it without asking. Either way it is gone when you quit.

    python start.py
"""
import getpass
import glob
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OSCAR = os.path.join(HERE, "oscar")
PROBLEMS_DIR = os.path.join(HERE, "problems")
RESULTS = os.path.join(HERE, "results")
RUNNER = os.path.join(HERE, "run_experiment.py")

# settings.txt is parsed in run_experiment.py so that both entry points read the file
# the same way and cannot drift apart.
try:
    from run_experiment import load_settings, DEFAULT_SETTINGS, SETTINGS_PATH
except ImportError:
    sys.stderr.write("run_experiment.py is not beside this file. Run start.py from the "
                     "package root: %s\n" % HERE)
    raise SystemExit(1)

CFG, CFG_COMPLAINTS = load_settings()

# Held in memory only. Nothing here is written to disk. The environment wins over
# settings.txt, which is what lets someone export a key and a model and skip the file.
SESSION = {
    "OSCAR_API_KEY": os.environ.get("OSCAR_API_KEY", ""),
    "OSCAR_API_BASE": os.environ.get("OSCAR_API_BASE", "") or CFG["api_base"],
    "OSCAR_SMALL_MODEL": os.environ.get("OSCAR_SMALL_MODEL", "") or CFG["small_model"],
    "OSCAR_LARGE_MODEL": os.environ.get("OSCAR_LARGE_MODEL", "") or CFG["large_model"],
}

# Median and 90th-percentile wall-clock in minutes, and how often OSCAR reached the
# optimum, over the paper's own runs: n=100 per cell, one one-shot repetition (the
# slower of the two models) and one OSCAR run at n=1. The distributions are long-tailed,
# so the pair is what to expect and not a bound.
TIMING = {              # problem: (1-shot median, p90, OSCAR median, p90, OSCAR % optimal)
    "buchheim2018": (9, 38, 95, 183, 100),
    "colombi2017": (8, 33, 41, 64, 72),
    "elci2022": (6, 22, 30, 57, 95),
    "letelier2022": (6, 22, 30, 137, 97),
    "mehrotra1996": (2, 7, 9, 12, 100),
}


def problems():
    return sorted(d for d in os.listdir(PROBLEMS_DIR)
                  if os.path.isdir(os.path.join(PROBLEMS_DIR, d)))


def child_env(problem=None):
    env = dict(os.environ)
    for k, v in SESSION.items():
        if v:
            env[k] = v
        else:
            env.pop(k, None)
    if problem:
        env["OSCAR_PROBLEM"] = os.path.join(PROBLEMS_DIR, problem)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def rule(title=""):
    print("\n" + "-" * 66)
    if title:
        print(title)
        print("-" * 66)


def ask(prompt, default=None, choices=None):
    while True:
        suffix = " [%s]" % default if default is not None else ""
        try:
            raw = input("%s%s: " % (prompt, suffix)).strip()
        except EOFError:
            return default
        if not raw and default is not None:
            raw = str(default)
        if not raw:
            continue
        if choices and raw not in choices:
            print("   choose one of: %s" % ", ".join(choices))
            continue
        return raw


def ask_int(prompt, default, lo=1, hi=1000):
    while True:
        raw = ask(prompt, str(default))
        try:
            v = int(raw)
        except ValueError:
            print("   a whole number, please")
            continue
        if not (lo <= v <= hi):
            print("   between %d and %d" % (lo, hi))
            continue
        return v


def pick_problem(default=None):
    ps = problems()
    default = default if default in ps else ps[0]
    print("")
    for i, p in enumerate(ps, 1):
        print("   %d) %-14s %s" % (i, p, "  <- settings.txt" if p == default else ""))
    while True:
        raw = ask("problem", str(ps.index(default) + 1))
        if raw.isdigit() and 1 <= int(raw) <= len(ps):
            return ps[int(raw) - 1]
        if raw in ps:
            return raw
        print("   pick a number from the list")


def optimum(problem):
    p = os.path.join(PROBLEMS_DIR, problem, "gurobi_solution", "tiny_solution.json")
    try:
        return json.load(io.open(p, encoding="utf-8")).get("objective_value")
    except Exception:
        return None


def span(minutes):
    if minutes < 90:
        return "%d minutes" % int(round(minutes))
    return "%.1f hours" % (minutes / 60.0)


def forecast(problem, n, method, nspec="1"):
    """What the paper's own runs took, so nobody starts a two-day job by accident."""
    t = TIMING.get(problem)
    if not t:
        return
    med, p90 = (t[0], t[1]) if method == "oneshot" else (t[2], t[3])
    unit = "repetition" if method == "oneshot" else "run"
    print("")
    print("   What to expect")
    print("     one %s of %s took about %d minutes in our runs;"
          % (unit, problem, med))
    print("     the slowest tenth took %d." % p90)
    if method == "oscar":
        print("     OSCAR reached the optimum on %d of our 100 runs here." % t[4])
    print("     %d of them: around %s, longer if your endpoint is busier than ours."
          % (n, span(n * med)))
    print("     nearly all of that is waiting -- the endpoint answering each call,")
    print("     and COPT solving each program the Coder writes.")
    if method == "oscar" and nspec != "1":
        try:
            mult = sum(int(x) for x in nspec.split(",")) / float(len(nspec.split(",")))
        except ValueError:
            mult = 1.0
        if mult > 1:
            print("     that figure is for n_per_level=1. At %s a run makes up to %.0f"
                  % (nspec, mult))
            print("     times as many attempts, so expect longer.")
    if problem == "buchheim2018":
        print("     this is the quadratic problem, the slowest of the five: the solve,")
        print("     not the model, is what takes the time.")
    print("     you can stop at any point with Ctrl-C. Whatever finished is kept,")
    print("     and starting again continues from there.")


# ------------------------------------------------------------------- settings

def settings_menu():
    rule("Settings")
    print("Everything but the key comes from")
    print("   %s" % SETTINGS_PATH)
    print("Edit it in any text editor, then press r here to reload it.")
    print("")
    print("   models      small     %s" % SESSION["OSCAR_SMALL_MODEL"])
    print("               large     %s" % SESSION["OSCAR_LARGE_MODEL"])
    print("   endpoint              %s" % (SESSION["OSCAR_API_BASE"] or "not set"))
    print("   API key               %s   (memory only, never in the file)"
          % ("set (hidden)" if SESSION["OSCAR_API_KEY"] else "NOT SET"))
    print("")
    print("   defaults    problem   %s" % CFG["problem"])
    print("               one-shot  %s repetition(s) with the %s model"
          % (CFG["oneshot_reps"], CFG["oneshot_model"]))
    print("               OSCAR     %s run(s), n_per_level %s"
          % (CFG["oscar_runs"], CFG["oscar_n_per_level"]))
    for c in CFG_COMPLAINTS:
        print("   !  %s" % c)
    print("")
    print("   OSCAR uses exactly two models and no more: a cheap one it tries first and")
    print("   a stronger one it escalates to. Name them in settings.txt.")
    print("")
    print("   1) enter the API key now (typing is hidden)")
    print("   2) forget the key")
    print("   3) where do I get a key, and what do I put in settings.txt?")
    print("   r) reload settings.txt")
    print("   b) back")
    c = ask("choice", "3" if not SESSION["OSCAR_API_KEY"] else "b",
            ["1", "2", "3", "r", "b"])
    if c == "1":
        print("\nPaste the key and press Enter. Nothing will appear as you type.")
        print("If you would rather not paste it here, quit and set OSCAR_API_KEY in")
        print("your shell instead; this tool picks it up automatically.")
        try:
            k = getpass.getpass("   key: ").strip()
        except (EOFError, KeyboardInterrupt):
            k = ""
        if k:
            SESSION["OSCAR_API_KEY"] = k
            print("   held in memory for this session.")
            # A key, an endpoint and a model name all have to agree, and finding out
            # they do not an hour into a run is the expensive way to learn it.
            if ask("\n   Try one very short call now to check it works? (y/n)",
                   "y") == "y":
                probe()
    elif c == "2":
        SESSION["OSCAR_API_KEY"] = ""
        print("   forgotten.")
    elif c == "3":
        key_help()
    elif c == "r":
        reload_settings()


def key_help():
    rule("Getting a key, and giving it to OSCAR")
    print("You need one. Every experiment here calls a language model, and no")
    print("credentials ship with this package.")
    print("")
    print("Three things have to agree: an OpenAI-compatible endpoint, a key that")
    print("works on it, and two model names spelled the way that endpoint spells")
    print("them. The first two are yours; the names go in settings.txt.")
    print("")
    print("   The paper used Qwen, served by Alibaba Cloud")
    print("   ---------------------------------------------------------------")
    print("   1. Open an account on Alibaba Cloud Model Studio (Bailian) and")
    print("      issue an API key there. It begins with  sk-")
    print("   2. Copy the OpenAI-compatible base URL from that console. At the")
    print("      time of writing it is")
    print("         https://dashscope.aliyuncs.com/compatible-mode/v1        Beijing")
    print("         https://dashscope-intl.aliyuncs.com/compatible-mode/v1   Singapore")
    print("      Take it from the console rather than from here: these move.")
    print("   3. Put the URL and the two model names in settings.txt")
    print("         api_base    = https://dashscope.aliyuncs.com/compatible-mode/v1")
    print("         small_model = %s" % DEFAULT_SETTINGS["small_model"])
    print("         large_model = %s" % DEFAULT_SETTINGS["large_model"])
    print("      Our own runs went through a private workspace deployment, so")
    print("      your console may spell the models a little differently. Use the")
    print("      names it gives you.")
    print("")
    print("   Any other OpenAI-compatible provider works the same way")
    print("   ---------------------------------------------------------------")
    print("         api_base    = https://api.openai.com/v1")
    print("         small_model = gpt-4o-mini")
    print("         large_model = gpt-4o")
    print("      The two must differ: a cheap model that tries first and a")
    print("      stronger one the Escalator moves up to.")
    print("")
    print("   Giving this program the key -- either way works")
    print("   ---------------------------------------------------------------")
    print("   at the prompt   option 1 on the previous screen. Held in memory")
    print("                   only: never written to disk, never printed, never")
    print("                   on a command line, and gone when you quit.")
    print("   in your shell   set OSCAR_API_KEY before starting and it is picked")
    print("                   up without being asked for:")
    print("                      macOS, Linux   export OSCAR_API_KEY=sk-...")
    print("                      Windows        setx OSCAR_API_KEY sk-...")
    print("                                     then open a new terminal")
    print("")
    print("   Do not put the key in settings.txt. Nothing else in that file is")
    print("   secret, which is what makes it safe to keep in version control.")
    print("")
    print("   Option 1 on the main menu checks all of this, and will make one")
    print("   very short model call to confirm the key works -- worth doing")
    print("   before committing an hour to a run.")


def reload_settings():
    global CFG, CFG_COMPLAINTS
    CFG, CFG_COMPLAINTS = load_settings()
    SESSION["OSCAR_API_BASE"] = os.environ.get("OSCAR_API_BASE", "") or CFG["api_base"]
    SESSION["OSCAR_SMALL_MODEL"] = os.environ.get("OSCAR_SMALL_MODEL", "") or CFG["small_model"]
    SESSION["OSCAR_LARGE_MODEL"] = os.environ.get("OSCAR_LARGE_MODEL", "") or CFG["large_model"]
    print("\n   reloaded: %s / %s, problem %s"
          % (SESSION["OSCAR_SMALL_MODEL"], SESSION["OSCAR_LARGE_MODEL"], CFG["problem"]))
    for c in CFG_COMPLAINTS:
        print("   !  %s" % c)


# ------------------------------------------------------------------- checking

CHECK = r'''
import io, json, os, sys
ok = True
try:
    import problem
    inst = json.load(io.open(problem.instance_path("tiny_instance.json"), encoding="utf-8"))
    ref = json.load(io.open(os.path.join(problem.PROBLEM_DIR, "gurobi_solution",
                                         "tiny_solution.json"), encoding="utf-8"))
    print("  problem data      OK   optimum %s" % problem.known_optimum("tiny_instance.json"))
except Exception as e:
    ok = False; print("  problem data      FAILED  %s: %s" % (type(e).__name__, e))
try:
    import build_simulator
    sim = build_simulator.load_certified_simulator()
    v = sim(inst, ref)
    good = v.get("feasible") if isinstance(v, dict) else v
    print("  Simulator         OK   accepts the reference solution: %s" % good)
except Exception as e:
    ok = False; print("  Simulator         FAILED  %s: %s" % (type(e).__name__, e))
try:
    g = problem.check_feasibility(inst, ref)
    print("  trusted checker   OK   agrees: %s" % (g.get("feasible") if isinstance(g, dict) else g))
except Exception as e:
    ok = False; print("  trusted checker   FAILED  %s: %s" % (type(e).__name__, e))
try:
    import coptpy, copt_env
    lic = copt_env.ensure_license(strict=False, verbose=False)
    print("  COPT              %s" % ("OK   licensed" if lic else
          "NOT LICENSED -- it would solve a smaller problem and every result would be wrong"))
    ok = ok and lic
except Exception as e:
    ok = False; print("  COPT              FAILED  %s: %s" % (type(e).__name__, e))
sys.exit(0 if ok else 1)
'''


def check_setup():
    rule("Checking your setup")
    p = pick_problem(CFG["problem"])
    print("")
    r = subprocess.run([sys.executable, "-c", CHECK], cwd=OSCAR, env=child_env(p))
    print("")
    print("  API key           %s" % ("set" if SESSION["OSCAR_API_KEY"] else
                                      "NOT SET -- no model can be called"))
    print("  endpoint          %s" % (SESSION["OSCAR_API_BASE"] or "NOT SET"))
    print("  models            %s / %s" % (SESSION["OSCAR_SMALL_MODEL"],
                                           SESSION["OSCAR_LARGE_MODEL"]))
    if not SESSION["OSCAR_API_KEY"] or not SESSION["OSCAR_API_BASE"]:
        print("")
        print("  Everything above the line needs no key -- that is how a missing key")
        print("  is told apart from a broken install. Running an experiment does need")
        print("  one: menu 2, then 3, says where to get it and where to put it.")
    if SESSION["OSCAR_API_KEY"] and SESSION["OSCAR_API_BASE"]:
        if ask("\nMake one very short model call to confirm the key works? (y/n)", "n") == "y":
            probe()
    print("")
    print("  setup is %s" % ("usable" if r.returncode == 0 else "NOT ready -- see above"))


PROBE = r'''
import os, sys
from utils import get_response
try:
    out = get_response("Reply with the single word: ok", os.environ["OSCAR_SMALL_MODEL"])
    print("  model call        OK   replied %r" % (str(out)[:40],))
except Exception as e:
    print("  model call        FAILED  %s: %s" % (type(e).__name__, str(e)[:160]))
    sys.exit(1)
'''


def probe():
    p = problems()[0]
    subprocess.run([sys.executable, "-c", PROBE], cwd=OSCAR, env=child_env(p))


# ---------------------------------------------------------------- experiments

def tally(problem, method, model=None):
    """(runs recorded, how many reached the optimum, mean tokens) so far."""
    ref = optimum(problem)
    n = ok = tok = 0
    if method == "oneshot":
        pat = os.path.join(RESULTS, problem, "oneshot", model or "*", "*", "summary.jsonl")
        for s in glob.glob(pat):
            for line in io.open(s, encoding="utf-8", errors="ignore"):
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                tok += r.get("total_tokens") or 0
                g = r.get("recomputed_objective")
                if r.get("gt_feasible") and g is not None and ref is not None \
                   and abs(float(g) - float(ref)) <= 1e-6 * max(1.0, abs(float(ref))):
                    ok += 1
    else:
        for f in glob.glob(os.path.join(RESULTS, problem, "oscar", "*", "run_*", "run_summary.json")):
            d = json.load(io.open(f, encoding="utf-8"))
            fin = d.get("final") or {}
            n += 1
            tok += d.get("total_tokens") or 0
            g = fin.get("gt_objective")
            if fin.get("gt_feasible") and g is not None and ref is not None \
               and abs(float(g) - float(ref)) <= 1e-4 * max(1.0, abs(float(ref))):
                ok += 1
    return n, ok, (tok / n if n else 0)


def run(cmd, problem, note, method, model=None, expect=""):
    print("")
    print("   %s" % note)
    if expect:
        print("   %s" % expect)
    print("   (your key is passed through the environment, not on this command line)")
    print("   $ %s" % " ".join(os.path.basename(c) if i == 1 else c
                               for i, c in enumerate(cmd)))
    print("")
    before = tally(problem, method, model)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE, env=child_env(problem)).returncode
    after = tally(problem, method, model)
    mins = (time.time() - t0) / 60.0

    new = after[0] - before[0]
    new_ok = after[1] - before[1]
    rule("Result")
    if rc != 0 and not new:
        print("   the run did not finish, and nothing was recorded.")
        print("   the output above says why; option 1 checks the usual causes.")
        return rc
    ref = optimum(problem)
    print("   %s on %s" % ("one-shot" if method == "oneshot" else "OSCAR", problem))
    print("   %d %s in %s"
          % (new, "repetition(s)" if method == "oneshot" else "run(s)", span(mins)))
    if new:
        print("   %d reached the optimum (%s) -- %.0f%%"
              % (new_ok, ref, 100.0 * new_ok / new))
        print("   about %.0f tokens each" % (after[2],))
    if after[0] > new:
        print("   %d in total for this problem so far, %d at the optimum (%.0f%%)"
              % (after[0], after[1], 100.0 * after[1] / after[0]))
    print("")
    print("   every attempt, the Simulator's verdicts and the cost per role are under")
    print("   %s" % os.path.join(RESULTS, problem, method))
    return rc


def which_model(default_key):
    """One of the two named models, and only those two."""
    s, l = SESSION["OSCAR_SMALL_MODEL"], SESSION["OSCAR_LARGE_MODEL"]
    print("")
    print("   s) %-28s the cheap model" % s)
    print("   l) %-28s the stronger one" % l)
    c = ask("model", "s" if default_key != "large" else "l", ["s", "l"])
    return s if c == "s" else l


def run_oneshot():
    rule("One-shot baseline")
    print("One Coder call plus syntax repair: no Simulator, no Reviewer, no second")
    print("attempt. This is what OSCAR is compared against.")
    print("Defaults are from settings.txt -- press Enter to take them.")
    p = pick_problem(CFG["problem"])
    reps = ask_int("repetitions", int(CFG["oneshot_reps"]), 1, 500)
    model = which_model(CFG["oneshot_model"])
    forecast(p, reps, "oneshot")
    print("")
    if ask("   start? (y/n)", "y") != "y":
        return
    cmd = [sys.executable, RUNNER, "--problem", p, "--method", "oneshot",
           "--model", model, "--reps", str(reps)]
    run(cmd, p, "%d one-shot repetition(s) on %s with %s" % (reps, p, model),
        "oneshot", model,
        expect="you will see one block per repetition, ending in ran_ok / gt_feasible / obj")


def run_oscar():
    rule("OSCAR")
    print("The full loop: a Simulator certifies each improvement, a Reviewer turns its")
    print("verdict into advice, and the Escalator decides which model tries next.")
    print("Defaults are from settings.txt -- press Enter to take them.")
    p = pick_problem(CFG["problem"])
    print("")
    print("Attempts at each configuration of the menu (Algorithm 1's n_j):")
    print("   1          one attempt at each   -- the paper's n=1")
    print("   1,1,3,3    three at the two dearer ones -- the paper's n=(1,1,3,3)")
    n = ask("attempts", CFG["oscar_n_per_level"])
    runs = ask_int("independent runs", int(CFG["oscar_runs"]), 1, 200)
    forecast(p, runs, "oscar", n)
    print("")
    if ask("   start? (y/n)", "y") != "y":
        return
    cmd = [sys.executable, RUNNER, "--problem", p, "--method", "oscar",
           "--runs", str(runs), "--n-per-level", n]
    run(cmd, p, "%d OSCAR run(s) on %s, n_per_level=%s" % (runs, p, n), "oscar",
        expect="you will see the structure call, the first formulation, then one line "
               "per attempt showing whether the Simulator certified it")


# ----------------------------------------------------------------- simulators

# Neither re-certifying the shipped Simulator nor building one from scratch is on this
# menu. Every run here uses the certified Simulator the paper used, so results are
# comparable with the published ones; swapping in a different oracle quietly makes them
# incomparable, and that is too easy to do by accident from a menu. The code and the
# data to build one are in the package -- oscar/generate_simulator.py,
# oscar/robustness.py, and each problem's certification examples -- and README.md says
# how.


# -------------------------------------------------------------------- results

def show_results():
    rule("Results so far")
    if not os.path.isdir(RESULTS):
        print("   nothing yet: %s does not exist" % RESULTS)
        return
    any_ = False
    for p in problems():
        ref = optimum(p)
        # one-shot
        for s in sorted(glob.glob(os.path.join(RESULTS, p, "oneshot", "*", "*", "summary.jsonl"))):
            rows = [json.loads(l) for l in io.open(s, encoding="utf-8", errors="ignore") if l.strip()]
            if not rows:
                continue
            any_ = True
            ok = sum(1 for r in rows
                     if r.get("gt_feasible") and r.get("recomputed_objective") is not None
                     and ref is not None
                     and abs(float(r["recomputed_objective"]) - float(ref)) <= 1e-6 * max(1.0, abs(float(ref))))
            model = os.path.basename(os.path.dirname(os.path.dirname(s)))
            print("   %-14s one-shot %-16s %3d run(s), %3d at the optimum (%3.0f%%)"
                  % (p, model, len(rows), ok, 100.0 * ok / len(rows)))
        # oscar
        runs = sorted(glob.glob(os.path.join(RESULTS, p, "oscar", "*", "run_*", "run_summary.json")))
        if runs:
            any_ = True
            ok = 0
            for f in runs:
                fin = (json.load(io.open(f, encoding="utf-8")).get("final") or {})
                g = fin.get("gt_objective")
                if fin.get("gt_feasible") and g is not None and ref is not None \
                   and abs(float(g) - float(ref)) <= 1e-4 * max(1.0, abs(float(ref))):
                    ok += 1
            print("   %-14s OSCAR    %-16s %3d run(s), %3d at the optimum (%3.0f%%)"
                  % (p, "", len(runs), ok, 100.0 * ok / len(runs)))
    if not any_:
        print("   nothing yet. Run an experiment first.")
    else:
        print("\n   full records, including every attempt and the cost per role, are under")
        print("   %s" % RESULTS)


# ----------------------------------------------------------------------- menu

MENU = """
OSCAR: Optimization modeling by Simulator, Coder, And Reviewer
A guided runner. Each of these is one run_experiment.py command, printed as it
goes, so this is also a way to learn the flags.

  setup
    1  check that your setup works
    2  settings and credentials      (key: %s, models: %s / %s)

  experiments
    3  run the one-shot baseline
    4  run OSCAR

  5  show results so far
  q  quit

What a run does -- the two models, the problem, the repetitions -- is set in
settings.txt beside this file. Expect to wait: a run is mostly the endpoint
answering and COPT solving. One OSCAR run took 9 minutes on the graph problem
in our own runs, which is the default, and 95 on the quadratic one.

Every run uses the certified Simulator from the paper, so your numbers are
comparable with the published ones. Building one from scratch is possible --
see "Building a Simulator" in README.md -- but it is not on this menu.
"""


def main():
    # Children write straight to the terminal while our own prints sit in a buffer,
    # so without this a heading can appear after the output it introduces.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if not os.path.isdir(OSCAR) or not os.path.isdir(PROBLEMS_DIR):
        print("Run this from the package root: %s" % HERE)
        return 1
    for c in CFG_COMPLAINTS:
        print("settings.txt: %s" % c)
    first = True
    while True:
        print(MENU % ("set" if SESSION["OSCAR_API_KEY"] else "not set",
                      SESSION["OSCAR_SMALL_MODEL"], SESSION["OSCAR_LARGE_MODEL"]))
        if first and not SESSION["OSCAR_API_KEY"]:
            print("No API key yet, and every experiment needs one. Choose 2, then 3:")
            print("it says where to get a key and where each piece of it goes.")
            print("")
        first = False
        try:
            c = ask("choice", "1", ["1", "2", "3", "4", "5", "q"])
        except KeyboardInterrupt:
            print("")
            return 0
        try:
            if c == "1":
                check_setup()
            elif c == "2":
                settings_menu()
            elif c == "3":
                run_oneshot()
            elif c == "4":
                run_oscar()
            elif c == "5":
                show_results()
            elif c == "q":
                SESSION["OSCAR_API_KEY"] = ""
                print("\nKey discarded. Nothing was written to disk.")
                return 0
        except KeyboardInterrupt:
            print("\n   stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
