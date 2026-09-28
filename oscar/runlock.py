"""
One owner per run directory.

Two processes writing into the same run directory silently interleave their
llm_metrics.jsonl and overwrite each other's attempts, and the result looks like
a plausible run rather than an error. That is easy to cause by accident -- a
relaunch while the previous process is still alive, a killed shell that left its
Python child running -- so the directory itself carries a lock.
"""

import contextlib
import errno
import os
import time


class RunDirBusy(RuntimeError):
    pass


def _alive(pid):
    """Best-effort liveness check that works on Windows without extra deps."""
    if pid <= 0:
        return False
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                               False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259           # STILL_ACTIVE
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError as e:
            return e.errno == errno.EPERM
        except Exception:
            return True                                  # assume busy if unsure


@contextlib.contextmanager
def run_dir_lock(run_dir, stale_after=7200):
    """Claim run_dir for this process, or raise RunDirBusy.

    A lock whose owner is gone (or that is older than stale_after) is taken over,
    so a crashed run does not block a rerun.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, ".owner")

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ("%d %f" % (os.getpid(), time.time())).encode())
            os.close(fd)
            break
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            try:
                with open(path, "r", encoding="utf-8") as f:
                    owner_pid = int(f.read().split()[0])
                age = time.time() - os.path.getmtime(path)
            except Exception:
                owner_pid, age = -1, stale_after + 1
            if owner_pid != os.getpid() and _alive(owner_pid) and age < stale_after:
                raise RunDirBusy(
                    "%s is owned by live process %d (started %.0fs ago). Another run "
                    "is using this directory; use a different run index, or stop that "
                    "process first." % (run_dir, owner_pid, age))
            try:
                os.remove(path)                          # stale: take it over
            except OSError:
                pass

    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
