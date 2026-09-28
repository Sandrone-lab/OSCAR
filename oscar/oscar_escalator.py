"""
OSCAR under the deterministic prior-free Escalator (Algorithm 1), for Section 5.1.

Menu. Two model sizes give 3m-2 = 4 configurations, ordered by cost. Cost is
dominated by the Coder, since the Coder emits a whole program while the Reviewer
emits a short diagnostic:

    a1 = (Reviewer S, Coder S)      cheapest
    a2 = (Reviewer L, Coder S)
    a3 = (Reviewer S, Coder L)
    a4 = (Reviewer L, Coder L)      dearest

with one attempt allowed at each level before escalating.

Preparations, rebuilt for every run and charged to the run: the structure
constructor, the Simulator constructor (built and certified against example
solutions from *other* instances), and one initial Coder call from the large
model, whose output becomes the first incumbent.

Improvement loop. A window starts with the incumbent fixed and walks the menu.
Attempts at the same Coder size revise the most recent candidate; escalating the
Coder discards the draft and reverts to the incumbent, so a large model is never
asked to repair a small model's formulation. A candidate is certified when the
Simulator finds it feasible and strictly better than the incumbent, which starts
a new window at a1. When a window exhausts all four levels with nothing
certified, the Escalator stops -- that is the only stopping rule.

The ground-truth checker is run alongside for scoring only. Its verdict is never
shown to the Reviewer or the Coder.

Usage:
  python oscar_escalator.py --instance data.json --runs 10
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

import our_model
from our_model import (
    generate_optimization_code, revise_code, generate_structure,
    format_hard_feedback, build_combined_review_feedback, safe_call_review,
)
from build_simulator import load_certified_simulator, charge_build_to
from grading import safe_check_feasibility, safe_simulate, recompute_objective
from copt_env import ensure_license, looks_unlicensed, solver_slot
from utils import summarize_metrics, call_role, set_call_role
from runlock import run_dir_lock, RunDirBusy

SMALL = os.environ.get("OSCAR_SMALL_MODEL", "qwen3.5-flash")
LARGE = os.environ.get("OSCAR_LARGE_MODEL", "qwen3.6-flash")
SIZE = {SMALL: 0, LARGE: 1}

# Optionally put a single frontier model in the two ROUTED roles -- Reviewer and
# Coder -- leaving repair and structure construction on the Qwen models, which
# Section 2.2 does not route.  With this set the menu collapses to one level:
# there is nothing to escalate to, so Algorithm 1 degenerates to plain retries,
# which is the point of the comparison.
MAIN_MODEL = os.environ.get("OSCAR_MAIN_MODEL", "").strip()
if MAIN_MODEL:
    SIZE[MAIN_MODEL] = 2

# Algorithm 1 with n_j = 1 at every level.
MENU = [
    {"name": "a1", "reviewer": SMALL, "coder": SMALL},
    {"name": "a2", "reviewer": LARGE, "coder": SMALL},
    {"name": "a3", "reviewer": SMALL, "coder": LARGE},
    {"name": "a4", "reviewer": LARGE, "coder": LARGE},
]
if MAIN_MODEL:
    MENU = [{"name": "a1", "reviewer": MAIN_MODEL, "coder": MAIN_MODEL}]

# Hold a run to the first k menu rows. With k=1 and n_per_level=1 a window
# is exactly one improvement attempt, which is what the feedback-detail
# test needs: the escalator itself must not be part of what is measured.
try:
    MENU_LEVELS = int(os.environ.get("OSCAR_MENU_LEVELS", "0"))
except ValueError:
    MENU_LEVELS = 0
if MENU_LEVELS > 0:
    MENU = MENU[:MENU_LEVELS]

# Algorithm 1's n_j: attempts allowed at each configuration before escalating.
# Either one number for every level, or one per level ("3,3,1,1").
def parse_n_per_level(spec):
    """Accept 2, "2", or "3,3,1,1"; return a list with one entry per menu level."""
    if isinstance(spec, (list, tuple)):
        vals = [int(v) for v in spec]
    else:
        text = str(spec).strip()
        vals = [int(v) for v in text.split(",")] if "," in text else [int(text)] * len(MENU)
    if len(vals) == 1:
        vals = vals * len(MENU)
    if len(vals) != len(MENU):
        raise ValueError("n_per_level needs 1 or %d values, got %d" % (len(MENU), len(vals)))
    if any(v < 0 for v in vals):
        raise ValueError("n_per_level values must be non-negative")
    return vals


def format_n_per_level(vals):
    vals = list(vals)
    return str(vals[0]) if len(set(vals)) == 1 else ",".join(str(v) for v in vals)


N_PER_LEVEL = os.environ.get("OSCAR_N_PER_LEVEL", "1")

# When does an attempt start from the incumbent rather than from the previous
# draft? "escalation" is the rule of Section 5: revert only when the Coder gets
# larger, so a large model never repairs a small one's formulation. "always"
# is the ablation: every attempt starts fresh from the incumbent, so attempts
# within a window are independent draws and nothing is ever revised.
REVERT_POLICY = os.environ.get("OSCAR_REVERT", "escalation")

# After the menu is exhausted and the incumbent is still infeasible, repeat the
# LAST menu row until it becomes feasible. Uncapped by default: a window that
# STARTS infeasible is a window in which a better state provably exists, so
# there is no principled budget on finding it -- the finite menu budget is an
# argument about improving a feasible incumbent, and it does not apply here.
# Set OSCAR_EXTEND_MAX=0 to switch the phase off and reproduce the earlier arms.
EXTEND_MAX = int(os.environ.get("OSCAR_EXTEND_MAX", "-1"))
# Drop the draft chain back to the incumbent every this many extension attempts,
# so the phase cannot become one unbounded revision of a broken formulation.
EXTEND_RESET = int(os.environ.get("OSCAR_EXTEND_RESET", "3"))
# How often to log progress of an uncapped extension. The phase is allowed to run
# without a cap, so this is the only signal that one is not converging.
EXTEND_WARN = int(os.environ.get("OSCAR_EXTEND_WARN", "50"))


def build_schedule(n_per_level):
    """The within-window call sequence: n_j attempts at each level, in cost order."""
    vals = parse_n_per_level(n_per_level)
    return [dict(level, rep=i + 1)
            for level, n in zip(MENU, vals)
            for i in range(n)]


REPAIR_MODEL = SMALL      # syntax repair is not routed (Section 2.2)
# ...except that a small model can stall on a solver API it does not know.
# After this many failed repair rounds the larger model takes over the repair.
# 0 restores the unrouted behaviour described in Section 2.2.
REPAIR_ESCALATE_AFTER = int(os.environ.get("OSCAR_REPAIR_ESCALATE_AFTER", "2"))

# A program that runs and writes a solution of the WRONG SHAPE is a bug of the
# same kind as one that writes nothing, so it is repaired rather than improved
# from. ON by default.
#
# Note what this does and does not cost. The repair happens inside the attempt,
# on REPAIR_MODEL -- the small model, unrouted, Section 2.2 -- so the Reviewer
# never sees it and no level of the menu is spent: a malformed file is a syntax
# bug, not a modelling judgement, and asking the Reviewer to weigh in on one
# would be asking the wrong question at the wrong price. bodur2017 measured it:
# 17 malformed repairs in a run and attempts-per-run still exactly 4.0.
#
# Off, the loop cannot say anything about shape at all. The verdict carries no
# violation count -- nothing was checked -- so violation progress can never
# fire, every attempt repeats the same unreadable file, and with an uncapped
# extension the run has no way to end. That is what stopped pecin2017 (74%
# malformed, archived) and the first belvaux2000 arm (70-97%, 40 attempts in one
# window, no exit).
REPAIR_MALFORMED = os.environ.get("OSCAR_REPAIR_MALFORMED", "1").strip() in ("1", "true", "yes")

# After this many failed repair rounds, ask the LARGE Reviewer once, and let its
# review drive the next round instead of the raw error text. 0 disables it.
#
# This escalates the KIND of help rather than the size of the model. The repair
# role may not touch the formulation -- it is told so -- so it can only ever
# patch the writer. Five rounds of that is strong evidence the writer is not the
# problem: belvaux2000's Coder wrote `production: list[0]`, an EMPTY decision
# set, which is what an over-constrained model produces and no amount of
# rewriting the output code will fix. The Reviewer is the role allowed to say
# the model is wrong.
REPAIR_REVIEW_AT = int(os.environ.get("OSCAR_REPAIR_REVIEW_AT", "5"))

# Stop repairing when the SAME failure comes back this many times, rather than
# after a fixed number of rounds.
#
# A fixed budget cannot tell progress from thrashing. ropke2009's Coder does not
# know the COPT API, and its rounds read
#     TypeError -> addVars kwarg -> COPT.NO_SOLUTION -> Envr.closeEnv
# -- four different errors, each genuinely fixed, and a budget of five cut it off
# mid-climb. Meanwhile 12 attempts elsewhere repeated ONE error twenty times.
#
# Measured over 2842 attempts with repair logs: 83% of the ones that eventually
# ran never repeat an error at all, and of the 32 that never ran, 12 repeated a
# single error the whole way. Stopping at 3 repeats ends 66% of those while
# costing 6% of the attempts that would have succeeded -- the same price as the
# fixed budget of 5, paid by the attempts that are actually stuck instead of the
# ones still making progress.
REPAIR_STALL_AT = int(os.environ.get("OSCAR_REPAIR_STALL_AT", "3"))


def _error_signature(msg):
    """What KIND of failure this is: exception class plus the head of its message.

    Two rounds that raise TypeError about different arguments are progress; two
    that raise the same one are not. The class alone is too coarse -- half of
    coptpy's misuse surfaces as AttributeError -- and the whole message too fine,
    since it often carries a line number that shifts on every rewrite.
    """
    import re
    m = re.findall(r"^\s*([A-Za-z_.]*(?:Error|Exception))\s*:?\s*(.*)$",
                   str(msg), re.M)
    if m:
        cls, rest = m[-1]
        return "%s: %s" % (cls.split(".")[-1], rest.strip()[:50])
    return (str(msg).strip().splitlines() or ["?"])[0][:60]


def build_repair_ladder(spec, main_model):
    """One entry per allowed repair round, saying which model debugs that round.

    "2,2,2" means two rounds on the small model, two on the large, two on the
    main model, and then the attempt is abandoned.  The ladder's LENGTH is the
    repair budget, which is the point: a program the cheap tier cannot fix must
    not be able to absorb twenty rounds of a model that will never fix it.

    Returns None when unset, which restores the single-threshold behaviour.
    """
    tiers = [SMALL, LARGE] + ([main_model] if main_model else [])
    try:
        counts = [int(x) for x in str(spec).split(",") if x.strip() != ""]
    except ValueError:
        counts = []
    if not counts:
        return None
    if len(counts) == 1:
        counts = counts * len(tiers)
    ladder = []
    for model, n in zip(tiers, counts):
        ladder.extend([model] * max(0, n))
    return ladder or None


REPAIR_LADDER = build_repair_ladder(os.environ.get("OSCAR_REPAIR_LADDER", ""), MAIN_MODEL)
STRUCT_MODEL = SMALL      # structure construction is reliable and cheap


# ---------------------------------------------------------------------------
# scoring helpers (never fed back into the loop)
# ---------------------------------------------------------------------------

# When nothing is feasible yet, "fewer violated constraints" is the only
# gradient available. On by default -- every arm has been run with it, including
# the baselines, so the comparison is between policies rather than between one
# arm that can see a gradient and one that cannot.
VIOLATION_PROGRESS = os.environ.get("OSCAR_VIOLATION_PROGRESS", "1").strip() in ("1", "true", "yes")

# Treat a violation-progress advance as a certification: end the window and open
# a new one from the improved incumbent, rather than continuing down the menu and
# stopping when it runs out. On by default: shedding a violated constraint IS
# the improvement available while the incumbent is infeasible, so a window that
# achieves one has done what a window is for, and should be rewarded with a
# fresh menu rather than spending the rest of this one.
# Set OSCAR_VP_CERTIFIES=0 to reproduce the earlier arms.
VP_CERTIFIES = os.environ.get("OSCAR_VP_CERTIFIES", "1").strip() in ("1", "true", "yes")


def violation_count(sim):
    """How many rules a simulator verdict reports broken, or None if unusable.

    Only an INFEASIBLE verdict carries a meaningful count. A malformed solution
    has no violation list because nothing was checked, and treating that as zero
    would make a program that produced no solution look closer to correct than
    one that nearly worked.
    """
    if not isinstance(sim, dict):
        return None
    if str(sim.get("status") or "").upper() != "INFEASIBLE":
        return None
    v = sim.get("violated_constraints")
    if not isinstance(v, list):
        return None
    # (how many distinct rules are broken, how many times in total).
    # Distinct first: that is what "fewer constraints violated" means, and it
    # cannot be gamed by a formulation that simply decides less. The instance
    # count only breaks ties between candidates breaking the same rules.
    return (len(set(v)), len(v))


def _objective_is_maximised():
    """The problem's own declared sense, defaulting to minimisation.

    Read through problem.py rather than from the environment so that the sense
    a run certifies against is the same one it is graded against.
    """
    try:
        import problem as _p
        return str(getattr(_p, "OBJECTIVE_SENSE", "min")).lower().startswith("max")
    except Exception:
        return False


_MAXIMISE = _objective_is_maximised()


def strictly_better(cand_obj, inc_obj, tol=1e-6):
    """An infeasible incumbent is beaten by any feasible candidate.

    The direction comes from the problem, not from an assumption. Every problem
    measured before colombi2017 minimises, so _MAXIMISE is False for all of them
    and this is the comparison they were run under.

    A solution whose reported objective is inconsistent is rejected outright by
    the Simulator (status INCONSISTENT_OBJECTIVE), so it never becomes an
    incumbent and there is nothing here to tie-break against.
    """
    if cand_obj is None:
        return False
    if inc_obj is None:
        return True
    if _MAXIMISE:
        return cand_obj > inc_obj + tol
    return cand_obj < inc_obj - tol


# ---------------------------------------------------------------------------
# one attempt: Coder call -> syntax repair -> run -> Simulator
# ---------------------------------------------------------------------------

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

def make_attempt(ctx, coder, history, attempt_dir, label):
    os.makedirs(attempt_dir, exist_ok=True)
    shutil.copy(ctx["instance_path"], os.path.join(attempt_dir, "data.json"))
    code_path = os.path.join(attempt_dir, "code.py")
    solution_path = os.path.join(attempt_dir, "solution.json")

    rec = {
        "label": label, "coder": coder, "debug_calls": 0, "ran_ok": False,
        "timed_out": False, "solution_present": False,
        "sim_feasible": None, "sim_objective": None, "sim_violated": None,
        "gt_feasible": None, "gt_violated": None, "gt_objective": None,
        "reported_objective": None, "error": None,
    }
    started = time.perf_counter()

    try:
        with call_role(ctx.get("coder_role", "coder"), label):
            code = generate_optimization_code(ctx["desc"], ctx["structure"], ctx["data_keys"],
                                              coder, history)
    except Exception as e:
        rec["error"] = "coder call failed: %s" % e
        rec["wall_sec"] = round(time.perf_counter() - started, 3)
        # A failed API call must cost this attempt, not the run. Returning None
        # here skipped the no-candidate fallback below and took the whole run
        # down on ``sim_result.get(...)``: ten of the twelve bodur2017 runs died
        # that way, and because the crash lands before the attempt record is
        # written, they left no evidence of why. An attempt whose Coder never
        # answered produced nothing, which the loop already knows how to handle.
        # DID_NOT_RUN rather than a status of its own: the Reviewer's handling of
        # it is already exactly right -- nothing was produced, so no rule of the
        # real problem has been shown broken -- and a status with no branch would
        # fall through to the generic "infeasible" text and tell the next Coder
        # to fix constraints that were never tested. The distinction that matters
        # for accounting is kept on the record instead.
        rec["coder_failed"] = True
        return rec, "", {"feasible": False, "status": "DID_NOT_RUN",
                         "objective_value": None, "no_candidate": True,
                         "violations": ["The Coder call did not return, so no "
                                        "program was written for this attempt: "
                                        "%s" % e]}
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    ladder = ctx.get("repair_ladder")
    # The ladder now schedules WHICH MODEL debugs each round; it no longer sets
    # the budget. A run that keeps fixing different errors should not be cut off
    # by the length of an escalation schedule -- the stall detector below is the
    # stopping rule, and max_debug only bounds the worst case.
    budget = ctx["max_debug"]
    for attempt in range(budget + 1):
        if os.path.exists(solution_path):
            os.remove(solution_path)
        try:
            with solver_slot():
                result = subprocess.run([sys.executable, RUN_CODE, "code.py"],
                                        cwd=attempt_dir, capture_output=True,
                                        text=True, timeout=ctx["proc_tl"])
        except subprocess.TimeoutExpired:
            rec["timed_out"] = True
            break
        if looks_unlicensed(result.stdout):
            rec["copt_unlicensed"] = True
        produced = (os.path.exists(solution_path)
                    and os.path.getsize(solution_path) > 0)
        malformed_why = None
        _sim = None
        if result.returncode == 0 and produced and REPAIR_MALFORMED and attempt < budget:
            # The Simulator is reachable here, so ask it now: a solution it
            # cannot read is a defect in the program's output, and this loop is
            # where defects are fixed. Its message names the field.
            try:
                with open(solution_path, "r", encoding="utf-8") as f:
                    _cand = json.load(f)
                _sim = safe_simulate(ctx["simulator"], ctx["instance"], _cand)
                if str((_sim or {}).get("status") or "").upper() == "MALFORMED_SOLUTION":
                    said = [str(v) for v in ((_sim or {}).get("violations") or [])][:4]
                    malformed_why = ("The program ran and wrote a solution, but it is not in "
                                     "the syntax the problem description specifies, so it "
                                     "cannot be judged at all. %s Fix how the solution is "
                                     "assembled and written -- the field names and the shape "
                                     "-- and do not change the model."
                                     % (" ".join(said) if said else ""))
            except Exception:
                malformed_why = None
        if result.returncode == 0 and produced and not malformed_why:
            rec["ran_ok"] = True
            break
        if attempt == budget:
            rec["ran_ok"] = bool(result.returncode == 0 and produced)
            rec["repair_exhausted"] = not rec["ran_ok"]
            break
        parts = []
        if result.stderr:
            parts.append("STDERR:\n" + result.stderr)
        if result.stdout:
            parts.append("STDOUT:\n" + result.stdout)
        error_message = "\n".join(parts) or ("Return code: %d" % result.returncode)
        if malformed_why:
            rec["malformed_repairs"] = rec.get("malformed_repairs", 0) + 1
            error_message = malformed_why + "\n\n" + error_message
        elif result.returncode == 0 and not produced:
            rec["empty_solution_repairs"] = rec.get("empty_solution_repairs", 0) + 1
            error_message = ("The program exited normally but wrote no solution.json, or wrote an empty one. Do not catch the exception and write an empty file: fix the underlying error so a complete solution is written.\n\n"
                             + error_message)
        # Stop when the same failure keeps coming back. A different error each
        # round means the repair is working and should be allowed to continue.
        _sig = _error_signature(error_message)
        _sigs = rec.setdefault("repair_signatures", [])
        _sigs.append(_sig)
        if REPAIR_STALL_AT and _sigs.count(_sig) >= REPAIR_STALL_AT:
            rec["ran_ok"] = False
            rec["repair_stalled_on"] = _sig
            rec["repair_exhausted"] = True
            break

        # Rounds of patching the writer have not worked, so stop assuming the
        # writer is the fault. The Reviewer is asked once -- the only role
        # permitted to say the FORMULATION is wrong -- and its answer drives the
        # next round instead of the raw error text.
        #
        # Only when a solution EXISTS. A program that will not run has produced
        # nothing for a reviewer to reason about, so it would read the same
        # traceback the repair model already has and answer the wrong question:
        # every such escalation on ropke2009 came back "hasModelingError: False,
        # the model accurately captures the PDPTW" -- correct, and useless, since
        # the fault was the solver API. It cost the last round of the budget.
        if REPAIR_REVIEW_AT and (attempt + 1) >= REPAIR_REVIEW_AT \
                and isinstance(_sim, dict) and not rec.get("repair_reviewed"):
            try:
                _verdict = _sim if isinstance(_sim, dict) else {
                    "feasible": False, "status": "DID_NOT_RUN",
                    "violations": ["The generated program could not be made to "
                                   "run: %s" % error_message[:400]]}
                with call_role("loop_reviewer", "repair_review"):
                    _rev = safe_call_review(ctx["desc"], code,
                                            hard_feedback_for(_verdict), LARGE)
                rec["repair_reviewed"] = True
                # Rendered the way a Coder normally reads a review, not as a
                # raw dict: the repair role is being handed the Reviewer's
                # output and should see it in the form it was written for.
                _txt = build_combined_review_feedback(
                    hard_feedback=hard_feedback_for(_verdict), llm_feedback=_rev)
                error_message = (
                    "%d repair rounds have not fixed this, so the fault may not "
                    "be in how the solution is written. A reviewer has looked at "
                    "the formulation itself; you MAY change the model if the "
                    "review says the model is what is wrong.\n\n%s\n\n%s"
                    % (attempt + 1, _txt, error_message))
            except Exception as e:
                rec["repair_review_failed"] = str(e)[:120]

        _log_repair_error(attempt_dir, attempt, error_message, code)
        try:
            if ladder:
                # Past the end of the schedule, stay on its dearest tier.
                repair_model = ladder[min(attempt, len(ladder) - 1)]
                rec["repair_escalated"] = repair_model != ladder[0]
            else:
                repair_model = REPAIR_MODEL
                if REPAIR_ESCALATE_AFTER and attempt >= REPAIR_ESCALATE_AFTER:
                    repair_model = LARGE
                    rec["repair_escalated"] = True
            rec.setdefault("repair_models", []).append(repair_model)
            with call_role(ctx.get("repair_role", "repair"), label):
                code = revise_code(ctx["desc"], ctx["structure"], ctx["data_keys"],
                                   model=repair_model, review_context=error_message,
                                   previous_code=code)
            rec["debug_calls"] += 1
        except Exception as e:
            rec["error"] = "repair call failed: %s" % e
            break
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

    rec["wall_sec"] = round(time.perf_counter() - started, 3)

    sim_result = None
    if rec["ran_ok"] and os.path.exists(solution_path) and os.path.getsize(solution_path) > 0:
        try:
            with open(solution_path, "r", encoding="utf-8") as f:
                solution = json.load(f)
            rec["solution_present"] = True
            rec["reported_objective"] = solution.get("objective_value")
            sim_result = safe_simulate(ctx["simulator"], ctx["instance"], solution)
            rec["sim_feasible"] = bool(sim_result.get("feasible"))
            rec["sim_violated"] = sim_result.get("violated_constraints")
            if rec["sim_feasible"]:
                obj = sim_result.get("objective_value")
                try:
                    rec["sim_objective"] = float(obj) if obj is not None else None
                except (TypeError, ValueError):
                    rec["sim_objective"] = None
            rec["sim_status"] = sim_result.get("status")
            gt = safe_check_feasibility(ctx["instance"], solution)
            rec["gt_feasible"] = gt["feasible"]
            rec["gt_violated"] = gt["violated_constraints"]
            rec["schema_error"] = gt.get("schema_error", False)
            rec["gt_objective"] = recompute_objective(ctx["instance"], solution)
        except Exception:
            rec["error"] = "grading failed: %s" % traceback.format_exc(limit=2)

    if sim_result is None:
        # Nothing was evaluated, so no rule of the real problem has been shown to
        # be violated. Reporting this as a constraint violation would be false and
        # sends the Reviewer hunting for bugs that are not there.
        if rec.get("timed_out"):
            why, cause = ("The generated program did not finish within the time limit, so no "
                          "candidate solution was produced.", "TIMEOUT")
        elif not rec.get("ran_ok"):
            why, cause = ("The generated program could not be made to run: the repair loop was "
                          "exhausted without producing an executable program.", "DID_NOT_RUN")
        else:
            why, cause = ("The generated program ran to completion, but the solver found no "
                          "feasible solution.", "NO_SOLUTION")
        sim_result = {"feasible": False, "status": cause, "objective_value": None,
                      "violations": [why], "no_candidate": True}
        rec["sim_status"] = cause
    return rec, code, sim_result


# How much of the Simulator's report the Reviewer and Coder are allowed to see.
# "full" is the framework as described; the other two exist to measure what the
# oracle contributes as a verdict versus what it contributes as an explanation.
FEEDBACK_LEVEL = os.environ.get("OSCAR_FEEDBACK_LEVEL", "full").strip().lower()

# No verdict at all. The wording must not imply the model is wrong -- that would
# be a verdict smuggled in as a hint -- nor that it is right. It asks for the
# same work the Reviewer does in every arm, with nothing external to go on.
_FEEDBACK_NONE = chr(10).join([
    "HARD_FEEDBACK:",
    "HARD_STATUS: NOT_AVAILABLE",
    "No checker has been run on the previous model, and no information about",
    "whether its solution satisfies the true problem constraints is available.",
    "Nothing is known about whether it is correct or incorrect.",
    "",
    "MANDATORY_REVISION_INSTRUCTION:",
    "Re-read the problem description and re-examine the previous model against",
    "it yourself. Decide on your own whether any rule of the real problem is",
    "modelled incorrectly, left out, or imposed where the description does not",
    "require it, and revise accordingly.",
])

# The verdict, and only the verdict.
_FEEDBACK_TERSE_BAD = chr(10).join([
    "HARD_FEEDBACK:",
    "HARD_STATUS: MODEL_REJECTED",
    "The previous model was checked against the true problem and REJECTED.",
    "No further information is available: you are told that it is wrong, but",
    "not which rule it breaks, not where, and not by how much.",
    "",
    "MANDATORY_REVISION_INSTRUCTION:",
    "Re-read the problem description and find the error yourself. The previous",
    "model is definitely wrong, so do not conclude that it is already correct.",
])


def format_rich_feedback(sim_result, max_rules=12, per_rule=3):
    """Hard feedback that also carries the NUMBERS the Simulator already computed.

    Every certified simulator returns violation_magnitudes alongside its
    messages -- {constraint, lhs, rhs, raw_excess, ratio} -- and the framework
    has been discarding them, sending the Coder "Capacity violation Fac 0 Scen 0"
    while holding "computed 22.0 against a limit of 10.0" in the same dict. The
    message says a rule broke; the magnitude says by how much, which is what
    tells a Coder whether a bound is slightly wrong or the constraint is missing.

    Two things this has to survive, both found in real simulator output:

    A flat list does not scale. One QWEN6 solution reports 126 violations, and
    pasting them all costs 13k characters to say what the worst few of each rule
    say better. Violations are grouped, and the groups are ordered by severity --
    NOT by constraint id, which on that same simulator is an edge index and
    sorts 0, 1, 10, 100.

    Not every magnitude is a magnitude. A boolean rule ("these two vertices share
    a colour") reports lhs = rhs = 1.0, and rendering that as "computed 1.0
    against a limit of 1.0" states a contradiction. The numbers are shown only
    when they actually demonstrate an excess.
    """
    msgs = sim_result.get("violations") or []
    mags = sim_result.get("violation_magnitudes") or []
    # Emitted in step by every simulator built so far, but a mismatch must
    # degrade to the plain message rather than attach the wrong numbers.
    paired = list(zip(msgs, mags)) if len(mags) == len(msgs) else [(m, None) for m in msgs]

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def severity(item):
        g = item[1] or {}
        for k in ("ratio", "raw_excess"):
            v = num(g.get(k))
            if v is not None:
                return v
        return 0.0

    def quantified(g):
        """True when the numbers say something the message does not."""
        if not g:
            return False
        lhs, rhs = num(g.get("lhs")), num(g.get("rhs"))
        return lhs is not None and rhs is not None and abs(lhs - rhs) > 1e-9

    groups = {}
    for m, g in paired:
        groups.setdefault((g or {}).get("constraint", "?"), []).append((m, g))

    ordered = sorted(groups.items(),
                     key=lambda kv: -max(severity(i) for i in kv[1]))

    lines = ["HARD_FEEDBACK:", "HARD_STATUS: MODEL_INFEASIBLE",
             "The solution produced by the previous generated model violates the true "
             "problem constraints.",
             "The following violations are detected by the deterministic feasibility "
             "checker, worst first, with the values it computed.",
             "These violations are ground truth and must not be ignored or contradicted "
             "by the LLM reviewer.",
             "", "VIOLATED_TRUE_CONSTRAINTS:"]

    shown_any_number = False
    for key, items in ordered[:max_rules]:
        items = sorted(items, key=severity, reverse=True)
        lines.append("Constraint %s -- %d violation(s):" % (key, len(items)))
        for m, g in items[:per_rule]:
            lines.append("  - %s" % m)
            if quantified(g):
                shown_any_number = True
                lines.append("      computed %s against a limit of %s (excess %s)"
                             % (g.get("lhs"), g.get("rhs"), g.get("raw_excess")))
        if len(items) > per_rule:
            lines.append("  ... and %d more violation(s) of this same constraint."
                         % (len(items) - per_rule))
    if len(ordered) > max_rules:
        lines.append("... and %d further constraint(s) violated, not listed."
                     % (len(ordered) - max_rules))

    lines += [
        "",
        "MANDATORY_REVISION_INSTRUCTION:",
        "Treat HARD_STATUS: MODEL_INFEASIBLE as a hard failure of the previous "
        "mathematical model.",
        "The next model must be changed so that the violated true constraints are "
        "explicitly respected.",
    ]
    # Deliberately NOT advising how to interpret the size of an excess. Any such
    # rule ("a large excess means the constraint is missing") is the author's
    # inference, not the oracle's finding, and it would make this arm a test of
    # hand-written guidance rather than of what the Simulator actually reports.
    lines.append(
        "This is not a syntax-only issue. Do not merely patch the code unless the "
        "violation is caused by output-format or variable-extraction errors.")
    return chr(10).join(lines)


def format_diagnose_feedback(sim_result, with_messages):
    """Feedback built from the Simulator's plain-language diagnoses.

    A diagnosis says which RULE OF THE REAL PROBLEM is broken, in the problem's
    own vocabulary, rather than which constraint index failed and by how much.
    """
    diags = sim_result.get("violation_diagnose") or []
    msgs = sim_result.get("violations") or []
    if not diags:
        # oracle predates the field -- fall back rather than send an empty report
        return format_hard_feedback({"feasible": False, "violations": msgs})
    lines = ["HARD_FEEDBACK:", "HARD_STATUS: MODEL_INFEASIBLE",
             "The solution produced by the previous generated model breaks rules of "
             "the real problem.",
             "These are ground truth and must not be ignored or contradicted by the "
             "LLM reviewer.", "", "RULES BROKEN:"]
    paired = (with_messages and len(msgs) == len(diags))
    for i, d in enumerate(diags[:12], 1):
        lines.append("%d. %s" % (i, str(d).strip()))
        if paired:
            lines.append("     (checker detail: %s)" % str(msgs[i - 1]).strip()[:110])
    if len(diags) > 12:
        lines.append("... and %d more." % (len(diags) - 12))
    lines += ["", "MANDATORY_REVISION_INSTRUCTION:",
              "Treat HARD_STATUS: MODEL_INFEASIBLE as a hard failure of the previous "
              "mathematical model.",
              "Change the model so that every rule listed above is respected.",
              "Do not delete or weaken a constraint to make the objective look better.",
              "This is not a syntax-only issue."]
    return chr(10).join(lines)


def hard_feedback_for(sim_result):
    """Turn a Simulator report into feedback text for the Reviewer.

    Deliberately problem-agnostic: it names the KIND of failure, never the
    constraints of any particular problem. The Reviewer and Coder already get the
    full problem description, so restating domain concepts here would both be
    redundant and tie the framework to one problem.
    """
    if FEEDBACK_LEVEL in ("fulldiag", "diagonly") \
            and not sim_result.get("feasible") \
            and str(sim_result.get("status") or "").upper() == "INFEASIBLE":
        return format_diagnose_feedback(
            sim_result, with_messages=(FEEDBACK_LEVEL == "fulldiag"))
    if FEEDBACK_LEVEL == "rich" and not sim_result.get("feasible")             and str(sim_result.get("status") or "").upper() == "INFEASIBLE":
        return format_rich_feedback(sim_result)
    if FEEDBACK_LEVEL == "none":
        return _FEEDBACK_NONE
    if FEEDBACK_LEVEL == "terse":
        # A feasible verdict already carries no diagnostics in the full arm --
        # it says "feasible, now improve it" and nothing more -- so passing it
        # through unchanged keeps the arms differing ONLY in what is said about
        # failures, which is the comparison being made.
        if sim_result.get("feasible"):
            return format_hard_feedback({"feasible": True})
        return _FEEDBACK_TERSE_BAD

    status = str(sim_result.get("status") or "").upper()
    warnings = sim_result.get("warnings") or []

    if sim_result.get("feasible"):
        text = format_hard_feedback({"feasible": True})

    elif status == "NO_SOLUTION":
        text = chr(10).join([
            "HARD_FEEDBACK:",
            "HARD_STATUS: NO_SOLUTION",
            "The generated program ran to completion, but the solver proved the model",
            "INFEASIBLE: no assignment of the decision variables satisfies all of the",
            "constraints as you wrote them. No candidate solution exists, so nothing",
            "could be checked against the true problem rules, and NO rule of the real",
            "problem has been shown to be violated.",
            "",
            "The problem instance itself is solvable. An infeasible model therefore",
            "means the formulation is over-constrained: it forbids solutions that the",
            "real problem allows.",
            "",
            "MANDATORY_REVISION_INSTRUCTION:",
            "Do not add or tighten constraints. Find what makes the model too tight.",
            "Look in particular for:",
            "  - a decision variable whose allowed domain or index range excludes",
            "    values the problem description permits, leaving it with no admissible",
            "    choice;",
            "  - a bound, limit or size derived from a quantity that is not given in",
            "    the data and was assumed or defaulted too small;",
            "  - an equality imposed where the description states an inequality, or a",
            "    constraint applied to more cases than the description requires.",
            "Before solving, verify that every decision variable has at least one",
            "admissible value under the constraints you have written.",
        ])

    elif status == "MALFORMED_SOLUTION":
        lines = [
            "HARD_FEEDBACK:",
            "HARD_STATUS: MALFORMED_SOLUTION",
            "The generated program wrote a solution file, but the file does not express",
            "a solution in the form the problem description requires, so it could not be",
            "evaluated at all. NO rule of the real problem has been shown to be violated,",
            "and the mathematical model may well be correct.",
            "",
            "What is wrong with the file:",
        ]
        for v in (sim_result.get("violations") or []):
            lines.append("  - %s" % v)
        lines += [
            "",
            "MANDATORY_REVISION_INSTRUCTION:",
            "Do not change the constraints or the objective. Fix the code that writes the",
            "solution file. Check that it runs after a solution has been found, that it",
            "reads the solved values of the decision variables rather than empty or",
            "default containers, and that it writes every field with the exact names and",
            "structure the problem description specifies.",
        ]
        text = chr(10).join(lines)

    elif status == "SIMULATOR_ERROR":
        text = chr(10).join([
            "HARD_FEEDBACK:",
            "HARD_STATUS: SIMULATOR_ERROR",
            (sim_result.get("violations") or ["The candidate could not be evaluated."])[0],
            "This is a failure of the evaluation, not a demonstrated fault in the model.",
            "",
            "MANDATORY_REVISION_INSTRUCTION:",
            "Keep the formulation, and make sure the solution file uses exactly the field",
            "names, types and structure the problem description specifies.",
        ])

    elif status in ("TIMEOUT", "DID_NOT_RUN"):
        text = chr(10).join([
            "HARD_FEEDBACK:",
            "HARD_STATUS: " + status,
            (sim_result.get("violations") or ["No candidate solution was produced."])[0],
            "No candidate solution exists, so nothing could be checked against the true",
            "problem rules, and NO rule of the real problem has been shown to be",
            "violated.",
            "",
            "MANDATORY_REVISION_INSTRUCTION:",
            ("Produce a formulation that solves within the time limit; a smaller or "
             "tighter encoding is usually the answer." if status == "TIMEOUT" else
             "Produce a program that runs. Keep the formulation simple and explicit."),
        ])

    elif status == "INCONSISTENT_OBJECTIVE":
        obj = sim_result.get("objective_value")
        lines = [
            "HARD_FEEDBACK:",
            "HARD_STATUS: INCONSISTENT_OBJECTIVE",
            "The solution produced by the previous generated model satisfies every rule",
            "of the real problem, so it could be executed exactly as it stands. It is",
            "NOT infeasible, and no constraint needs to be added or tightened.",
            "",
            "What is wrong is the objective value the model reports:",
        ]
        for v in (sim_result.get("violations") or []):
            lines.append("  - %s" % v)
        if obj is not None:
            lines.append("")
            lines.append("The solution actually achieves an objective value of %s." % obj)
        lines += [
            "",
            "MANDATORY_REVISION_INSTRUCTION:",
            "Do not rewrite the constraints. Find why the value the model reports",
            "disagrees with the solution it produced. Check that the quantity written",
            "into the solution file is the objective defined in the problem description,",
            "computed from the same data and constants; that any quantity derived in",
            "preprocessing and used in that computation is itself correct; and that no",
            "additional term present in the solver's internal objective is included in",
            "the reported value.",
        ]
        text = chr(10).join(lines)

    else:
        text = format_hard_feedback({"feasible": False,
                                     "violations": sim_result.get("violations", [])})

    if warnings:
        text += chr(10) + chr(10) + "NON_FATAL_WARNINGS:" + chr(10)
        for i, w in enumerate(warnings, 1):
            text += "%d. %s" % (i, w) + chr(10)
    return text


# ---------------------------------------------------------------------------
# one OSCAR run
# ---------------------------------------------------------------------------

def oscar_run(run_dir, instance_path, desc, examples, max_debug, proc_tl,
              max_windows=50, verbose=True, n_per_level=None, revert_policy=None):
    with run_dir_lock(run_dir):
        return _oscar_run(run_dir, instance_path, desc, examples, max_debug,
                          proc_tl, max_windows, verbose,
                          N_PER_LEVEL if n_per_level is None else n_per_level,
                          REVERT_POLICY if revert_policy is None else revert_policy)


def _oscar_run(run_dir, instance_path, desc, examples, max_debug, proc_tl,
               max_windows, verbose, n_per_level, revert_policy):
    os.environ["LLM_METRICS_PATH"] = os.path.join(run_dir, "llm_metrics.jsonl")

    with open(instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    log = {"instance": os.path.basename(instance_path),
           "n_per_level": format_n_per_level(parse_n_per_level(n_per_level)),
           "revert_policy": revert_policy,
           "preparations": {}, "windows": [], "stopped_because": None}
    t0 = time.perf_counter()

    # --- preparations, charged to this run ---------------------------------
    if verbose:
        print("[prep] structure constructor")
    with call_role("prep_structure"):
        structure = generate_structure(desc, model=STRUCT_MODEL)
    with open(os.path.join(run_dir, "structure.json"), "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2)

    # The Simulator is built once by build_simulator.py and shared by every run
    # ("once built, the oracle is fixed", Section 2.1). Its cost is still charged
    # here, so a run is priced as though it had built its own.
    if verbose:
        print("[prep] loading the certified simulator (built once, cost charged per run)")
    try:
        simulator_fn = load_certified_simulator()
    except Exception as e:
        log["stopped_because"] = "no certified simulator: %s" % e
        log["wall_sec"] = round(time.perf_counter() - t0, 3)
        return log
    n_charged = charge_build_to(os.environ["LLM_METRICS_PATH"])
    log["preparations"]["simulator"] = {"shared": True, "charged_calls": n_charged}

    ctx = {
        "desc": desc, "structure": structure, "instance": instance,
        "instance_path": instance_path, "data_keys": list(instance.keys()),
        "simulator": simulator_fn,
        "max_debug": max_debug, "proc_tl": proc_tl,
        "repair_ladder": REPAIR_LADDER,
    }

    # --- initial incumbent: one shot from the large Coder -------------------
    if verbose:
        print("[prep] initial formulation (Coder = %s)" % (MAIN_MODEL or LARGE))
    ctx["coder_role"], ctx["repair_role"] = "prep_initial_coder", "prep_initial_repair"
    rec, code, sim_result = make_attempt(ctx, MAIN_MODEL or LARGE, "",
                                         os.path.join(run_dir, "attempt_000_initial"), "initial")
    ctx["coder_role"], ctx["repair_role"] = "loop_coder", "loop_repair"
    incumbent = {"code": code, "sim": sim_result, "obj": rec.get("sim_objective"),
                 "feasible": bool(rec.get("sim_feasible")), "rec": rec}
    log["preparations"]["initial"] = rec
    if verbose:
        print("       feasible=%s obj=%s" % (incumbent["feasible"], incumbent["obj"]))

    # --- improvement loop ---------------------------------------------------
    n_attempt = 1
    for window_idx in range(max_windows):
        wlog = {"window": window_idx, "attempts": [], "certified_at": None}
        # Type is fixed at window start: infeasible here means a better state is
        # known to exist, which is what justifies an uncapped last stair.
        window_started_infeasible = not incumbent["feasible"]
        wlog["window_type"] = 1 if window_started_infeasible else 2
        draft = None            # most recent candidate in this window
        history = ""            # attempt history since the last revert
        prev_coder = None

        schedule = build_schedule(n_per_level)
        n_ext = 0                      # attempts spent in the extension phase
        sched_i = 0
        while sched_i < len(schedule):
            level = schedule[sched_i]
            sched_i += 1
            # Reset the draft chain after EXTEND_RESET revisions, not before the
            # first: the phase keeps revising a4's output, and the revert only
            # bounds how long that chain may grow.
            if level.get("ext") and EXTEND_RESET > 0 and n_ext > 0 \
                    and (n_ext % EXTEND_RESET) == 0:
                draft, history = None, ""
                if verbose:
                    print("  extension: %d attempt(s) in -> back to the incumbent"
                          % n_ext)
            if revert_policy == "always":
                draft, history = None, ""      # every attempt starts from the incumbent
            elif prev_coder is not None and SIZE[level["coder"]] > SIZE[prev_coder]:
                draft, history = None, ""      # revert only when the Coder escalates
                if verbose:
                    print("  escalating Coder -> reverting to incumbent")

            base = draft if draft is not None else incumbent
            if verbose:
                print("  window %d | %s#%d (Rev=%s, Coder=%s) | base=%s" % (
                    window_idx, level["name"], level["rep"], level["reviewer"],
                    level["coder"], "draft" if draft is not None else "incumbent"))

            hard = hard_feedback_for(base["sim"])
            with call_role("loop_reviewer", level["name"]):
                llm_review = safe_call_review(desc, base["code"], hard, level["reviewer"])
            combined = build_combined_review_feedback(hard_feedback=hard, llm_feedback=llm_review)
            history += "\nATTEMPT:\n\nPREVIOUS_GENERATED_CODE:\n```python\n%s\n```\n\n%s\n" % (
                base["code"], combined)

            _vals = parse_n_per_level(n_per_level)
            tag = (level["name"] if max(_vals) == 1
                   else "%s#%d" % (level["name"], level["rep"]))
            adir = os.path.join(run_dir, "attempt_%03d_%s" % (n_attempt, tag.replace("#", "_")))
            os.makedirs(adir, exist_ok=True)
            try:
                with open(os.path.join(adir, "review.json"), "w", encoding="utf-8") as f:
                    json.dump({"config": level["name"], "rep": level["rep"],
                               "reviewer": level["reviewer"],
                               "base": "draft" if draft is not None else "incumbent",
                               "hard_feedback": hard, "review": llm_review},
                              f, indent=2, default=str)
            except Exception:
                pass
            rec, code, sim_result = make_attempt(ctx, level["coder"], history, adir, tag)
            rec.update({"window": window_idx, "config": level["name"],
                        "level_rep": level["rep"],
                        "reviewer": level["reviewer"],
                        "base": "draft" if draft is not None else "incumbent"})
            n_attempt += 1

            cand_obj = rec.get("sim_objective") if rec.get("sim_feasible") else None
            rec["sim_status"] = sim_result.get("status")
            certified = strictly_better(cand_obj,
                                        incumbent["obj"] if incumbent["feasible"] else None)
            rec["certified"] = certified
            wlog["attempts"].append(rec)
            with open(os.path.join(adir, "attempt.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            if verbose:
                print("       ran_ok=%s sim=%s obj=%s certified=%s" % (
                    rec["ran_ok"], rec.get("sim_status") or rec["sim_feasible"],
                    cand_obj, certified))

            if certified:
                incumbent = {"code": code, "sim": sim_result, "obj": cand_obj,
                             "feasible": True, "rec": rec}
                wlog["certified_at"] = level["name"]
                break

            if VIOLATION_PROGRESS and not certified and not incumbent["feasible"]:
                cv = violation_count(sim_result)
                iv = violation_count(incumbent.get("sim"))
                rec["violations_now"] = list(cv) if cv else None
                rec["violations_incumbent"] = list(iv) if iv else None
                if cv is not None and iv is not None and cv < iv:
                    incumbent = {"code": code, "sim": sim_result, "obj": None,
                                 "feasible": False, "rec": rec}
                    rec["violation_progress"] = True
                    if VP_CERTIFIES:
                        wlog["certified_at"] = level["name"]
                        rec["certified_by"] = "violation_progress"
                        if verbose:
                            print("       fewer violations -> counted as "
                                  "certification, opening a new window")
                        break
                    if verbose:
                        print("       still infeasible, but %d rule(s)/%d "
                              "violation(s) vs %d/%d -> new incumbent"
                              % (cv[0], cv[1], iv[0], iv[1]))

            if not rec.get("ran_ok"):
                # The repair ladder gave up: this candidate never produced a
                # solution, so it is not a draft to build on.  Carrying it
                # forward would hand the next Coder a program already known to
                # be broken, and keep growing the history that made it broken.
                # Start the next attempt from the incumbent instead.
                draft, history = None, ""
                if verbose:
                    print("       repair exhausted -> reverting to incumbent")
            else:
                draft = {"code": code, "sim": sim_result, "obj": cand_obj,
                         "feasible": bool(rec.get("sim_feasible"))}
            prev_coder = level["coder"]

            if level.get("ext"):
                n_ext += 1
            # Extend whenever the window has certified nothing -- whether the
            # incumbent is infeasible (looking for fewer violations, or for
            # feasibility) or already feasible (looking for a better objective).
            # A certification breaks the loop above, so reaching here means the
            # window has not advanced the incumbent at all.
            # ...and only in a Type 1 window. Starting infeasible is what makes
            # a better state known to exist, which is what justifies searching
            # without a cap; a Type 2 window has no such guarantee and the menu
            # is its whole budget.
            if (EXTEND_MAX != 0 and sched_i >= len(schedule)
                    and window_started_infeasible
                    and wlog["certified_at"] is None
                    and (EXTEND_MAX < 0 or n_ext < EXTEND_MAX)):
                schedule.append(dict(MENU[-1], rep=1, ext=True))
                if EXTEND_WARN > 0 and n_ext and n_ext % EXTEND_WARN == 0:
                    print("       WARNING: %d extension attempts without "
                          "certifying (incumbent %s, obj %s)"
                          % (n_ext,
                             "feasible" if incumbent["feasible"] else "infeasible",
                             incumbent["obj"]), flush=True)

        wlog["extension_attempts"] = n_ext
        log["windows"].append(wlog)
        if wlog["certified_at"] is None:
            log["stopped_because"] = "escalator exhausted the menu without certifying"
            break
    else:
        log["stopped_because"] = "hit max_windows safety cap"

    log["final"] = {
        "sim_feasible": incumbent["feasible"],
        "sim_objective": incumbent["obj"],
        "gt_feasible": incumbent["rec"].get("gt_feasible"),
        "gt_objective": incumbent["rec"].get("gt_objective"),
        "gt_violated": incumbent["rec"].get("gt_violated"),
    }
    with open(os.path.join(run_dir, "incumbent_code.py"), "w", encoding="utf-8") as f:
        f.write(incumbent["code"])
    log["wall_sec"] = round(time.perf_counter() - t0, 3)
    log.update(summarize_metrics(os.environ["LLM_METRICS_PATH"]))
    return log


def main():
    ensure_license()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="data.json")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--start-run", type=int, default=0)
    ap.add_argument("--max-debug", type=int, default=20)
    ap.add_argument("--proc-tl", type=int, default=int(os.environ.get("OSCAR_PROC_TL", 600)))
    ap.add_argument("--max-windows", type=int, default=50, help="safety cap only")
    ap.add_argument("--revert", default=REVERT_POLICY, choices=["escalation", "always"],
                    help="when an attempt starts from the incumbent instead of the draft")
    ap.add_argument("--n-per-level", default=N_PER_LEVEL,
                    help="Algorithm 1 n_j: one number for all levels (\"2\") or one per "
                         "level in cost order (\"3,3,1,1\")")
    import problem as _pp
    _default_out = (os.path.join(BASE_DIR, "run_oscar") if _pp.PROBLEM_DIR == BASE_DIR
                    else os.path.join(BASE_DIR, "results", _pp.NAME, "run_oscar"))
    ap.add_argument("--outdir", default=_default_out)
    args = ap.parse_args()

    import problem as _p
    instance_path = (_p.instance_path(args.instance)
                     if not os.path.exists(os.path.join(BASE_DIR, args.instance))
                     else os.path.join(BASE_DIR, args.instance))
    stem = os.path.splitext(os.path.basename(args.instance))[0]
    out_root = os.path.join(args.outdir, stem)
    os.makedirs(out_root, exist_ok=True)

    import problem
    with open(problem.DESC_PATH, "r", encoding="utf-8") as f:
        desc = f.read()
    with open(os.path.join(problem.PROBLEM_DIR, "certification_examples.json"),
              "r", encoding="utf-8") as f:
        examples = json.load(f)

    summary_path = os.path.join(out_root, "summary.jsonl")
    for run in range(args.start_run, args.runs):
        print("=" * 70)
        print("OSCAR run %d/%d on %s" % (run + 1, args.runs, args.instance))
        run_dir = os.path.join(out_root, "run_%02d" % run)
        log = oscar_run(run_dir, instance_path, desc, examples,
                        args.max_debug, args.proc_tl, args.max_windows,
                        n_per_level=args.n_per_level, revert_policy=args.revert)
        log["run"] = run
        with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
        fin = log.get("final", {})
        print("  -> %s | gt_feasible=%s gt_obj=%s | %d windows | tokens=%s" % (
            log["stopped_because"], fin.get("gt_feasible"), fin.get("gt_objective"),
            len(log["windows"]), log.get("total_tokens")))

    print("\nSummary written to %s" % summary_path)


if __name__ == "__main__":
    main()
