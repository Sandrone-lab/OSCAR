"""
Wrapper used to execute a generated code.py.

The generated program decides its own solver parameters, so under concurrency it
would happily grab every core. This wrapper patches coptpy before the program
runs, forcing every model it creates to a bounded thread count. Whatever the
generated code sets afterwards is re-applied on solve(), so the cap holds even
if the program sets Threads itself.

It caps MEMORY the same way. A generated model is unbounded in size -- an integer
variable declared with no upper bound lets branch-and-bound grow its tree without
limit -- and nothing else on the machine stops it. One colombi2017 program reached
4.4 GB and pushed the box to 206 MB available and thousands of hard page faults a
second; a probe solve of a tiny model beside it then failed outright with
"(MEMORY) Fail to solve problem". Every solve next to a runaway fails for a reason
that has nothing to do with its own formulation, and the failure is recorded
against it. COPT's own MemLimit -- in megabytes; tripping it returns status 12
cleanly rather than raising -- confines the damage to the program that caused it.

Invoked as:  python run_code.py code.py
with cwd set to the attempt directory, exactly where code.py expects to find
data.json and to write solution.json.
"""

import os
import runpy
import sys

THREADS = int(os.environ.get("OSCAR_SOLVER_THREADS", 1))

# COPT reports a solve stopped at MemLimit as status 12. This coptpy build gives the
# value no name -- dir(COPT) runs 9 UNFINISHED, 10 INTERRUPTED, 11 ITERLIMIT, then
# jumps to 16 and 20 -- so it is matched by number.
MEMLIMIT_STATUS = 12


def _physical_mb():
    try:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MEMSTAT()
        s.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.ullTotalPhys / (1024.0 * 1024.0)
    except Exception:
        pass
    return 8192.0


def _default_memlimit_mb():
    """Physical memory less a reserve, divided among the solver slots.

    copt_env lets MAX_SOLVERS programs be inside COPT at once, so if each is held to
    this, all of them together cannot exhaust the machine. On this box: 8 GB, 2 GB
    kept back for the workers, the OS and the harness, four slots -> about 1.5 GB,
    some seven times the largest legitimate tiny-instance solve seen tonight.
    """
    solvers = int(os.environ.get("OSCAR_MAX_SOLVERS",
                                 max(1, (os.cpu_count() or 4) - 2)))
    return max(512.0, (_physical_mb() - 2048.0) / max(1, solvers))


# A solve may hold more than one of copt_env's slots (OSCAR_SOLVE_WEIGHT) and then gets
# that many slots' share of memory, so what all slots can claim together is unchanged.
# buchheim2018's exact quadratic model needs about 2.2 GB, more than one share. With the
# default weight of 1 the limit is exactly what it has always been.
try:
    SOLVE_WEIGHT = max(1, int(os.environ.get("OSCAR_SOLVE_WEIGHT") or 1))
except ValueError:
    SOLVE_WEIGHT = 1

MEMLIMIT_MB = float(os.environ.get("OSCAR_SOLVER_MEMLIMIT_MB") or _default_memlimit_mb() * SOLVE_WEIGHT)

# Where the Coder is not told to set a solver time limit (OSCAR_SOLVER_TL=0), the solver
# is stopped here instead, this many seconds in -- short of the escalator's process
# limit, so the program is still alive to write its incumbent. It never raises a limit
# the program set itself, and 0, the default, leaves every solve untouched.
try:
    SOLVER_TIME_CAP = float(os.environ.get("OSCAR_SOLVER_TIME_CAP") or 0)
except ValueError:
    SOLVER_TIME_CAP = 0.0

# COPT reports whether a MIP solution exists as a value, model.hasmipsol (0 or 1).
# Coders keep calling it -- model.hasmipsol() -- which raises "TypeError: 'int'
# object is not callable" after the solve has finished, so the answer is lost on the
# program's last lines. By 2026-09-11 that error had appeared in 428 of 12,030
# attempts; repair usually removed the call, but 23 attempts ended without running on
# it, and colombi2017/n2 run_84 spent four hours regenerating it. With
# OSCAR_HASMIPSOL_SHIM=1 the value can also be called: model.hasmipsol and
# model.hasmipsol() both give 0 or 1, and nothing else changes. run_variant_arm.py
# turns it on for arms from 2026-09-11 and records it in arm_settings.json; unset --
# the default -- leaves COPT's own behaviour.
HASMIPSOL_SHIM = os.environ.get("OSCAR_HASMIPSOL_SHIM", "0").strip().lower() in ("1", "true", "yes")


class _CallableInt(int):
    """An int that answers a call with itself, so model.hasmipsol() works too."""

    def __call__(self):
        return int(self)


def _shim_hasmipsol(cls):
    if getattr(cls, "_oscar_hasmipsol_shim", False):
        return
    lookup = cls.__getattr__        # coptpy resolves hasmipsol, status, objval ... here

    def hasmipsol(self):
        return _CallableInt(lookup(self, "hasmipsol"))

    try:
        cls.hasmipsol = property(hasmipsol)
        cls._oscar_hasmipsol_shim = True
    except (AttributeError, TypeError):
        pass                        # not patchable in this build: COPT's behaviour stands


# MemLimit governs only what COPT allocates inside solve(). A program can exhaust the
# machine before it ever gets there: colombi2017/n1 run_03 enumerated every subset of
# its 29 non-depot nodes -- itertools.combinations over range(1, 29) -- to write one
# cut per subset, some 2^29 constraints, and reached 6.8 GB of private memory in
# Python while still BUILDING the model. So the process also polices itself. A
# watchdog thread reads its own private bytes twice a second; coptpy releases the GIL
# inside solve (a probe thread ticked 292 times during a 3 s solve), so it covers the
# build and the solve alike. The limit sits above MemLimit, so a solve that hits
# COPT's own limit still stops cleanly first.
PROCESS_LIMIT_MB = float(os.environ.get("OSCAR_PROCESS_MEMLIMIT_MB") or (MEMLIMIT_MB + 384.0))


def _private_mb():
    """This process's private bytes in MB, or None where it cannot be read."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC),
                                           wintypes.DWORD]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return pmc.PrivateUsage / (1024.0 * 1024.0)
    return None


def _start_memory_watchdog():
    import threading
    import time
    if PROCESS_LIMIT_MB <= 0:
        return
    try:
        if _private_mb() is None:
            return                      # cannot measure here; leave it to MemLimit
    except Exception:
        return

    def watch():
        while True:
            try:
                mb = _private_mb()
            except Exception:
                return
            if mb is not None and mb > PROCESS_LIMIT_MB:
                try:
                    sys.stderr.write(
                        "OSCAR: the program was stopped after using %d MB of memory, over "
                        "its %d MB limit, while building or solving the model. Something is "
                        "being created in exponential or very large numbers. The usual cause "
                        "is a constraint or variable written for every subset or combination "
                        "of an index set (itertools.combinations, powerset loops), which for "
                        "a few dozen nodes is millions of constraints. Use a formulation whose "
                        "size grows polynomially with the instance -- for example a flow "
                        "formulation rather than one cut per subset -- and give every integer "
                        "variable an upper bound.\n" % (mb, PROCESS_LIMIT_MB))
                    sys.stderr.flush()
                finally:
                    os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watch, name="oscar-memory-watchdog", daemon=True).start()


def _patch_coptpy():
    try:
        import coptpy
    except ImportError:
        return  # let the generated program produce its own import error
    COPT = coptpy.COPT
    if HASMIPSOL_SHIM:
        _shim_hasmipsol(coptpy.Model)

    def _cap(model):
        for name, value in (("Threads", THREADS),
                            ("BarThreads", THREADS),
                            ("SimplexThreads", THREADS),
                            ("MipTasks", THREADS),
                            ("MemLimit", MEMLIMIT_MB)):
            try:
                model.setParam(getattr(COPT.Param, name), value)
            except Exception:
                pass  # parameter absent in this COPT build
        if SOLVER_TIME_CAP > 0:
            try:
                current = model.getParam(COPT.Param.TimeLimit)
            except Exception:
                current = None
            if current is None or current > SOLVER_TIME_CAP:
                try:
                    model.setParam(COPT.Param.TimeLimit, SOLVER_TIME_CAP)
                except Exception:
                    pass

    orig_create = coptpy.Envr.createModel

    def createModel(self, *a, **k):
        model = orig_create(self, *a, **k)
        _cap(model)
        if HASMIPSOL_SHIM:
            _shim_hasmipsol(type(model))    # a subclass, should createModel return one
        cls = type(model)
        if not getattr(cls, "_oscar_solve_patched", False):
            orig_solve = cls.solve

            def solve(self, *sa, **sk):
                _cap(self)          # re-assert after the program's own setParam calls
                out = orig_solve(self, *sa, **sk)
                # Say WHY, where the repair loop will read it. Without this a model
                # stopped at the limit looks like any other empty solve, and the
                # repair model is left to guess.
                try:
                    if self.status == MEMLIMIT_STATUS:
                        sys.stderr.write(
                            "OSCAR: the solver stopped at its memory limit of %d MB%s. "
                            "The model is too large to solve here. Give every integer "
                            "variable an upper bound -- an integer with no bound lets "
                            "the search tree grow without limit -- and do not create "
                            "variables or constraints for index combinations the "
                            "problem does not use.\n"
                            % (MEMLIMIT_MB, "" if getattr(self, "hasmipsol", 0)
                               else " before finding any solution"))
                        sys.stderr.flush()
                except Exception:
                    pass
                return out

            cls.solve = solve
            cls._oscar_solve_patched = True
        return model

    coptpy.Envr.createModel = createModel


def main():
    if len(sys.argv) < 2:
        print("usage: run_code.py <script.py>", file=sys.stderr)
        return 2
    script = sys.argv[1]
    _start_memory_watchdog()
    _patch_coptpy()
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
