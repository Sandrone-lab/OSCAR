"""
Make sure COPT runs licensed.

Without a license COPT starts "with size limitations for non-commercial use" and
still returns answers, so an unlicensed run does not fail -- it silently solves a
different, smaller problem. That would distort every measurement in the study,
so the drivers call ensure_license() before making a single LLM call.

The license lives in <repo>/COPT (license.dat + license.key). Setting
COPT_LICENSE_DIR in this process makes every subprocess that runs a generated
code.py inherit it.
"""

import contextlib
import errno
import os
import random
import subprocess
import sys
import tempfile
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LICENSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "COPT"))

PROBE = """
import coptpy as cp
env = cp.Envr()
m = env.createModel("probe")
x = m.addVars(20, vtype="B", nameprefix="x")
m.setObjective(cp.quicksum(x[i] for i in range(20)), sense=cp.COPT.MINIMIZE)
m.solve()
"""

UNLICENSED_MARKERS = ("No license found", "size limitations")

# Point COPT at the licence as soon as this module is imported, not only when a
# driver's main() runs. Anything that imports the harness -- a driver, a notebook,
# a test -- then inherits it, and no subprocess can end up size-limited by
# accident. The strict probe still runs in main(); this is just the env var.
os.environ.setdefault("COPT_LICENSE_DIR", DEFAULT_LICENSE_DIR)


def license_dir():
    return os.environ.get("COPT_LICENSE_DIR") or DEFAULT_LICENSE_DIR


def ensure_license(strict=True, verbose=True):
    """Point COPT at the license and verify a solve runs licensed.

    Returns True when licensed. With strict=True an unlicensed COPT raises,
    because silently size-limited solves invalidate the whole experiment.
    """
    d = license_dir()
    os.environ["COPT_LICENSE_DIR"] = d

    if not os.path.exists(os.path.join(d, "license.dat")):
        msg = "No license.dat in COPT_LICENSE_DIR=%s" % d
        if strict:
            raise RuntimeError(msg + " -- COPT would run size-limited and every "
                                     "measurement would be wrong.")
        if verbose:
            print("WARNING: " + msg)
        return False

    proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                          text=True, timeout=120)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        msg = "COPT probe failed (exit %d): %s" % (proc.returncode, out.strip()[-400:])
        if strict:
            raise RuntimeError(msg)
        if verbose:
            print("WARNING: " + msg)
        return False
    if any(m in out for m in UNLICENSED_MARKERS):
        msg = ("COPT started unlicensed despite COPT_LICENSE_DIR=%s -- it is running "
               "with size limitations, so results would be meaningless." % d)
        if strict:
            raise RuntimeError(msg)
        if verbose:
            print("WARNING: " + msg)
        return False

    if verbose:
        print("COPT licensed (COPT_LICENSE_DIR=%s)" % d)
    return True


def looks_unlicensed(text):
    """True if a generated program's output shows COPT ran size-limited."""
    return bool(text) and any(m in text for m in UNLICENSED_MARKERS)



# ---------------------------------------------------------------------------
# Solver admission control.
#
# The study is LLM-latency-bound: a single attempt spends minutes waiting on the
# API and seconds to a few minutes in COPT. So we want many runs in flight at
# once but only a few solves, or the box thrashes and every solve gets slower
# (which also corrupts any wall-clock number we report).
#
# Two knobs, deliberately separate:
#   OSCAR_MAX_SOLVERS - how many generated programs may be inside COPT at once
#   OSCAR_SOLVER_THREADS - threads each of those may use
# Keep MAX_SOLVERS * SOLVER_THREADS at or below the physical core count.
#
# The semaphore is a directory of exclusively-created slot files, so it works
# across independent driver processes on Windows without extra dependencies.
# ---------------------------------------------------------------------------

MAX_SOLVERS = int(os.environ.get("OSCAR_MAX_SOLVERS", max(1, (os.cpu_count() or 4) - 2)))
SOLVER_THREADS = int(os.environ.get("OSCAR_SOLVER_THREADS", 1))
SLOT_DIR = os.environ.get("OSCAR_SLOT_DIR") or os.path.join(
    tempfile.gettempdir(), "oscar_solver_slots")


def _reap_stale(stale_after):
    """Drop slots whose owner died without releasing them."""
    now = time.time()
    try:
        names = os.listdir(SLOT_DIR)
    except OSError:
        return
    for name in names:
        p = os.path.join(SLOT_DIR, name)
        try:
            if now - os.path.getmtime(p) > stale_after:
                os.remove(p)
        except OSError:
            pass


@contextlib.contextmanager
def solver_slot(max_solvers=None, stale_after=1800, poll=(0.25, 1.5), verbose=False,
                weight=None, exclusive=None):
    """Block until the solve may run, then hold its slots for the duration.

    weight -- how many of the MAX_SOLVERS slots the solve holds (OSCAR_SOLVE_WEIGHT,
    default 1). run_code.py scales the solve's memory limit by the same number, so a
    solve that needs more than one slot's share of memory takes it from the
    machine-wide budget: while a weight-2 solve runs, two fewer solves fit beside it.
    buchheim2018's exact quadratic model needs about 2.2 GB, over one share.

    exclusive -- a lock name (OSCAR_EXCLUSIVE_SOLVE, default none). Solves sharing it
    run one at a time however many slots are free. The lock is taken first, so solves
    of the same kind queue for it instead of each holding half its slots.

    With the defaults this takes exactly one slot, as it always has. Falls through
    immediately when max_solvers <= 0, so a single-process run pays nothing.
    """
    n = MAX_SOLVERS if max_solvers is None else max_solvers
    if n <= 0:
        yield None
        return
    if weight is None:
        try:
            weight = int(os.environ.get("OSCAR_SOLVE_WEIGHT") or 1)
        except ValueError:
            weight = 1
    weight = max(1, min(int(weight), n))
    if exclusive is None:
        exclusive = os.environ.get("OSCAR_EXCLUSIVE_SOLVE", "").strip()

    os.makedirs(SLOT_DIR, exist_ok=True)
    token = ("%d-%f" % (os.getpid(), time.time())).encode()

    def take(path):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as e:
            if e.errno not in (errno.EEXIST, errno.EACCES):
                raise
            return False
        os.write(fd, token)
        os.close(fd)
        return True

    waited, lock, slots = 0.0, None, []
    try:
        if exclusive:
            name = "exclusive_" + "".join(c if c.isalnum() else "_" for c in exclusive)
            path = os.path.join(SLOT_DIR, name)
            while not take(path):
                _reap_stale(stale_after)
                d = random.uniform(*poll)
                time.sleep(d)
                waited += d
            lock = path
        while len(slots) < weight:
            got = False
            for i in range(n):
                p = os.path.join(SLOT_DIR, "slot_%02d" % i)
                if p not in slots and take(p):
                    slots.append(p)
                    got = True
                    break
            if not got:
                _reap_stale(stale_after)
                d = random.uniform(*poll)
                time.sleep(d)
                waited += d
        if verbose and waited:
            print("  waited %.1fs for a solver slot" % waited)
        yield slots[0]
    finally:
        for p in slots + ([lock] if lock else []):
            try:
                os.remove(p)
            except OSError:
                pass


def clear_slots():
    """Remove every slot file. Use only when nothing is running."""
    try:
        for name in os.listdir(SLOT_DIR):
            os.remove(os.path.join(SLOT_DIR, name))
    except OSError:
        pass


if __name__ == "__main__":
    ensure_license()
    print("max concurrent solvers: %d x %d thread(s)  (cpu_count=%s)"
          % (MAX_SOLVERS, SOLVER_THREADS, os.cpu_count()))
    print("slot dir: %s" % SLOT_DIR)
