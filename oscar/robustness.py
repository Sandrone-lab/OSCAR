"""
Robustness battery for a candidate Simulator.

A certification set of hand-written examples tests the shapes someone thought
of. Three times now a simulator certified on such a set while a branch it never
reached was broken -- most recently a typo that raised NameError on an empty
decision list, a shape the Coder produces routinely.

So the shapes are generated instead of imagined: take a known-good solution and
mutate it every way the schema allows something to be missing or wrong. The
simulator must reject each one as MALFORMED_SOLUTION and must never raise.

Case names are derived from problem.py's schema description, so this battery
follows the problem rather than being written for one.
"""

import copy

import problem

MALFORMED = "MALFORMED_SOLUTION"


def _same_kind(existing, replacement):
    """Would substituting this value not actually be a type violation?

    The member mutations exist to test that a simulator rejects a member of the
    WRONG type. Where the decisions are a list of records that works: a bare
    number in place of a record is malformed. Where the decisions are a list of
    numbers -- a binary selection vector, say -- "first member is number"
    replaces a number with a number. That is a perfectly well formed solution
    that merely selects something else, so the simulator answers INFEASIBLE, the
    battery demands MALFORMED_SOLUTION, and the same case fails on every
    revision until the build gives up. bertsimas2022 lost a whole build to it.

    Booleans are numbers in Python and are treated as such here, since a
    solution encoding a binary decision as true/false is the same shape.
    """
    def kind(v):
        if v is None:
            return "null"
        if isinstance(v, bool) or isinstance(v, (int, float)):
            return "number"
        if isinstance(v, str):
            return "string"
        if isinstance(v, list):
            return "list"
        if isinstance(v, dict):
            return "object"
        return "other"
    return kind(existing) == kind(replacement)


def _carries_decisions(good, instance, f):
    """Does removing this field change what the TRUSTED checker says?

    The only dependable test of whether a field carries a decision, and it is
    the checker's own answer rather than a guess from the field's name.
    """
    probe = copy.deepcopy(good)
    if f not in probe:
        return False
    probe.pop(f)
    try:
        r = problem.check_feasibility(instance, probe)
    except Exception:
        return True          # the checker cannot proceed without it
    return not (isinstance(r, dict) and r.get("feasible"))


def _empty_is_legal(good, instance, field, value):
    """Is emptying this collection a legal DECISION rather than a malformed input?

    The convention is that a decision collection which is empty expresses no
    decision, so the simulator should call it MALFORMED_SOLUTION. That holds
    wherever every item must be handled -- a packing, a covering, an assignment.

    It does not hold for a prize-collecting problem, where choosing nothing is a
    choice. colombi2017's hauler may serve no profitable arc at all, and the
    trusted checker accepts that with a net profit of zero. Demanding the
    simulator reject it asks it to throw away a solution the Coder legitimately
    produced -- and it did: 14 of 74 real solutions rejected, every one of them
    an empty tour the checker calls feasible.

    Emptying a collection also changes the objective those decisions achieve, so
    the checker's first complaint is consistency. That is not the question being
    asked, so the objective is restated before the verdict is read.
    """
    probe = copy.deepcopy(good)
    probe[field] = value
    try:
        r = problem.check_feasibility(instance, probe)
    except Exception:
        return False
    if isinstance(r, dict) and r.get("feasible"):
        return True
    cur = probe.get("objective_value")
    for m in ((r or {}).get("violation_magnitudes") or []):
        try:
            if cur is None or abs(float(m.get("lhs")) - float(cur)) <= 1e-9:
                probe["objective_value"] = float(m.get("rhs"))
                r2 = problem.check_feasibility(instance, probe)
                return bool(isinstance(r2, dict) and r2.get("feasible"))
        except (TypeError, ValueError):
            continue
    return False


def _order_is_free(reordered, instance):
    """Does the TRUSTED checker still accept the solution after reordering?

    The only reliable way to tell a set from a positional vector. If the checker
    still passes it, order carried nothing and demanding the simulator tolerate
    the reordering is a real requirement; if it now fails, order was identity
    and the demand would be asking the simulator to be wrong.
    """
    try:
        r = problem.check_feasibility(instance, reordered)
    except Exception:
        return False
    return bool(isinstance(r, dict) and r.get("feasible"))


def _primary_field(good, instance=None):
    """The field that carries the decisions, for a problem with no single list.

    Taking the first declared field that is not the objective picks whatever the
    schema happens to list first, and solution schemas lead with metadata as
    often as with decisions. On bertsimas2022 that chose `instance_id` -- an
    instance NAME -- and the battery then demanded the simulator reject a
    solution for omitting it. The simulator rightly refused to, the same six
    cases failed on every revision, and the build gave up: no oracle, for a
    problem that was never at fault.

    So ask the trusted checker instead. A field whose removal leaves the verdict
    unchanged is not a decision, and requiring a simulator to reject solutions
    over it is a demand to be wrong. Without an instance to probe with there is
    nothing to ask, and the old order-based guess stands.
    """
    candidates = [f for f in problem.SOLUTION_FIELDS
                  if f != "objective_value"
                  and f not in problem.SOLUTION_SCALAR_FIELDS
                  and f in good]
    if instance is not None:
        for f in candidates:
            if _carries_decisions(good, instance, f):
                return f
    return candidates[0] if candidates else None


def _generic_cases(good, instance=None):
    """Mutations for a solution whose decisions are not one list of uniform entries.

    Only the field carrying the decisions is required: without it there is
    nothing to check, so the simulator must say so rather than judge. Optional
    and derived fields are left alone, since rejecting a solution for those
    would be over-strict.
    """
    field = _primary_field(good, instance)
    if not field:
        return []
    cases = []
    s = copy.deepcopy(good); s.pop(field, None)
    cases.append(("missing '%s'" % field, s, MALFORMED))
    for label, val in [("string", "nope"), ("null", None), ("number", 3),
                       ("empty list", []), ("empty object", {})]:
        # An empty collection is malformed only where choosing nothing is not a
        # choice. Where it is, the checker accepts it and this demand would make
        # the simulator reject a legal solution.
        if (label.startswith("empty") and instance is not None
                and _empty_is_legal(good, instance, field, val)):
            continue
        s = copy.deepcopy(good); s[field] = val
        cases.append(("'%s' is %s" % (field, label), s, MALFORMED))
    holder = good.get(field)
    if isinstance(holder, list) and holder:
        for label, val in [("string", "x"), ("number", 1), ("null", None)]:
            if _same_kind(holder[0], val):
                continue
            s = copy.deepcopy(good)
            s[field] = [val] + list(copy.deepcopy(holder)[1:])
            cases.append(("first member of '%s' is %s" % (field, label), s, MALFORMED))
    elif isinstance(holder, dict) and holder:
        k0 = next(iter(holder))
        for label, val in [("string", "x"), ("null", None), ("list", [1])]:
            if _same_kind(holder[k0], val):
                continue
            s = copy.deepcopy(good)
            s[field] = dict(copy.deepcopy(holder)); s[field][k0] = val
            cases.append(("member '%s' of '%s' is %s" % (k0, field, label), s, MALFORMED))
    return cases


def malformed_cases(good, instance=None):
    """(name, solution, expected_status) for inputs that are not valid solutions."""
    field = problem.SOLUTION_LIST_FIELD
    entry_fields = problem.SOLUTION_ENTRY_FIELDS
    cases = []

    if not field or field not in good:
        # No single designated list of decisions: follow the declared schema instead.
        for name, val in [("not an object: list", []), ("not an object: string", "nope"),
                          ("not an object: number", 7), ("not an object: null", None)]:
            cases.append((name, val, MALFORMED))
        cases.append(("empty object", {}, MALFORMED))
        return cases + _generic_cases(good, instance)

    # the container itself is not an object
    for name, val in [("not an object: list", []), ("not an object: string", "nope"),
                      ("not an object: number", 7), ("not an object: null", None)]:
        cases.append((name, val, MALFORMED))
    cases.append(("empty object", {}, MALFORMED))

    # the decision collection is missing or unusable
    s = copy.deepcopy(good); s.pop(field, None)
    cases.append(("missing '%s'" % field, s, MALFORMED))
    for label, val in [("empty list", []), ("string", "nope"), ("object", {}),
                       ("number", 3), ("null", None)]:
        s = copy.deepcopy(good); s[field] = val
        cases.append(("'%s' is %s" % (field, label), s, MALFORMED))

    # entries are not objects
    for label, val in [("string", "x"), ("number", 1), ("list", []), ("null", None)]:
        s = copy.deepcopy(good); s[field] = [val] + list(s[field][1:])
        cases.append(("first entry is %s" % label, s, MALFORMED))

    # each required field of an entry, missing in turn
    for f in entry_fields:
        s = copy.deepcopy(good)
        s[field] = copy.deepcopy(s[field])
        s[field][0] = {k: v for k, v in s[field][0].items() if k != f}
        cases.append(("entry missing '%s'" % f, s, MALFORMED))

    # each required field of an entry, wrong type
    for f in entry_fields:
        for label, val in [("string", "abc"), ("null", None), ("list", [1])]:
            s = copy.deepcopy(good)
            s[field] = copy.deepcopy(s[field])
            s[field][0] = dict(s[field][0]); s[field][0][f] = val
            cases.append(("entry '%s' is %s" % (f, label), s, MALFORMED))

    # required scalars missing or wrong type
    for f in problem.SOLUTION_SCALAR_FIELDS:
        s = copy.deepcopy(good); s.pop(f, None)
        cases.append(("missing '%s'" % f, s, MALFORMED))
        for label, val in [("string", "abc"), ("null", None), ("list", [1])]:
            s = copy.deepcopy(good); s[f] = val
            cases.append(("'%s' is %s" % (f, label), s, MALFORMED))

    return cases


def tolerance_cases(good, good_objective, instance=None):
    """Inputs that are unusual but VALID; the simulator must not reject them.

    Over-strict rejection is as damaging as a crash: it throws away correct
    answers, which is how an earlier simulator cost a whole experimental arm.
    """
    cases = []
    s = copy.deepcopy(good); s["an_unrelated_extra_field"] = {"note": "ignore me"}
    cases.append(("extra unknown top-level field", s, "OK"))

    field = problem.SOLUTION_LIST_FIELD
    if not field or field not in good or not isinstance(good.get(field), list)             or not good[field] or not isinstance(good[field][0], dict):
        # The decisions are not one list of uniform entries, so the entry-level
        # tolerances below do not apply. Reordering may or may not be harmless,
        # and which one it is cannot be told from the members' type:
        #
        #   tran2018's `stations` is a list of chosen node ids. It is a SET --
        #   order carries nothing, and a simulator that rejects a reordering is
        #   over-strict, which is what this case exists to catch.
        #
        #   bertsimas2022's `x` is 31 portfolio weights indexed by security. The
        #   index IS the identity, so reversing it hands every weight to a
        #   different security. Demanding "OK" there asks the simulator to
        #   accept a solution it should reject, and the build spends its whole
        #   budget failing a case that cannot be satisfied.
        #
        # Both are lists of plain numbers, so ask the trusted checker instead:
        # if the reordered solution still passes, order carries nothing and the
        # tolerance is real.
        for f in problem.SOLUTION_FIELDS:
            v = good.get(f)
            if not (isinstance(v, list) and len(v) > 1):
                continue
            t = copy.deepcopy(good)
            t[f] = list(reversed(t[f]))
            if instance is not None and not _order_is_free(t, instance):
                continue
            cases.append(("'%s' in a different order" % f, t, "OK"))
            break
        return cases

    s = copy.deepcopy(good)
    s[problem.SOLUTION_LIST_FIELD] = copy.deepcopy(s[problem.SOLUTION_LIST_FIELD])
    s[problem.SOLUTION_LIST_FIELD][0] = dict(s[problem.SOLUTION_LIST_FIELD][0])
    s[problem.SOLUTION_LIST_FIELD][0]["extra_entry_field"] = 1
    cases.append(("extra unknown field inside an entry", s, "OK"))
    if good_objective is not None:
        s = copy.deepcopy(good)
        s[problem.SOLUTION_LIST_FIELD] = list(reversed(s[problem.SOLUTION_LIST_FIELD]))
        cases.append(("entries in a different order", s, "OK"))
    return cases


def run_battery(fn, instance, good, good_objective=None, verbose=True):
    """Run every case against fn. Returns (ok, failures)."""
    failures = []
    cases = ([(n, s, e, "malformed") for n, s, e in malformed_cases(good, instance)] +
             [(n, s, e, "tolerance") for n, s, e in tolerance_cases(good, good_objective, instance)])
    for name, sol, expect, kind in cases:
        try:
            res = fn(instance, sol)
        except Exception as e:
            failures.append((name, kind, "raised %s: %s" % (type(e).__name__, str(e)[:80])))
            if verbose:
                print("    %-44s RAISED %s" % (name[:44], type(e).__name__))
            continue
        if not isinstance(res, dict):
            failures.append((name, kind, "returned %s, not a dict" % type(res).__name__))
            continue
        got = str(res.get("status") or "").upper()
        if kind == "malformed":
            bad = (res.get("feasible") is not False) or got != expect
        else:
            bad = (res.get("feasible") is not True)
        if bad:
            failures.append((name, kind, "feasible=%s status=%s" % (res.get("feasible"), got or "missing")))
            if verbose:
                print("    %-44s got feasible=%s status=%s" % (name[:44], res.get("feasible"), got or "missing"))
    if verbose:
        print("    battery: %d case(s), %d failure(s)" % (len(cases), len(failures)))
    return not failures, failures
