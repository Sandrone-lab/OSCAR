import os
import json
import subprocess
import time
from utils import get_response, call_role
import simulator
import importlib
def extract_code_block(text):
    """从LLM响应中提取代码块"""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + len("```")
        end = text.find("```", start)
        return text[start:end].strip()
    return text.strip()

def revise_code(desc, structure, data_keys, model, review_context="", previous_code=""):
    """Step 2: 基于数据结构生成完整的求解代码"""
    prompt = f"""
You are an expert in revising code with COPT (coptpy).

PROBLEM DESCRIPTION:
{desc}

DATA STRUCTURE (parameters and their shapes):
{json.dumps(structure, indent=2)}

The actual data file "run_dev/data.json" contains the following parameters (keys):
{json.dumps(data_keys, indent=2)}

The actual data will be in a JSON file at "data.json" with exactly this structure.
All parameter values will be loaded from this file.

Here is the error message from the previous code:

{review_context}

Here is the previous code:

{previous_code}

You are revising the existing code based on the error message. You may only modify the code to make it runnable; you should not modify the model. You can only revise the wrong part and NEVER modify the model.

CODE FORMAT:
Return ONLY the complete Python code in a markdown code block.
"""
    response = get_response(prompt, model)
    return extract_code_block(response)

def generate_simulator(description,model):
    prompt_simulator = f"""
You are an expert reviewer for operations research problem feasibility check.
You are generating a function for a specific OR problem to check the feasibility.
Problem description:
-----
{description}
-----
NOTE: the description provides necessary variables, your task is generating a function to check the feasibility. The input is a solution and the output is true or false. If true, it means that the solution is feasible

The form of your function should be: feasibility_check(data,solution)

IMPORTANT: If the solution contains an "objective_value" field, you MUST verify that the recomputed objective value matches the reported value. Include this as a constraint check (e.g., constraint index for objective validation) and report any mismatch as a violation with appropriate LHS/RHS values.

IMPORTANT: IMPORTANT: Distinguish three different reasons for rejecting a solution, and label them differently.

(a) MALFORMED_SOLUTION - the input does not express a decision in the syntax the problem description specifies: a required field is missing or misnamed, a collection that should hold the decisions is empty or is not a list, an entry is missing one of its required fields, or a value has the wrong type. You cannot judge such an input at all. Set "feasible" to False, set "status" to "MALFORMED_SOLUTION", say exactly which part of the required syntax is missing or wrong, and do NOT report any rule of the problem as violated -- nothing has been checked. Do this check FIRST, before any other, and be tolerant of extra fields the description does not mention.

(b) INFEASIBLE - the solution is well formed, but breaks one or more of the rules stated in the problem description above, so it could not be carried out. Set "feasible" to False, set "status" to "INFEASIBLE", and describe the rule that is violated.

(c) INCONSISTENT_OBJECTIVE - the solution is well formed and every rule stated in the problem description holds, so the solution could be carried out, but the reported "objective_value" differs from the value you recompute by more than 0.5. This is NOT a rule violation and must never be described as one. Set "feasible" to False so the solution is still rejected, set "status" to "INCONSISTENT_OBJECTIVE", and write a message saying the solution itself is valid while the reported objective is inconsistent, naming both numbers, for example: "Solution is valid, but the reported objective_value 9.0011 is inconsistent with the recomputed objective 3.0." Do NOT list any rule as violated in this case. A difference of 0.5 or less is expected, because the problem description may not fix the value of every constant appearing in the objective, and is not an inconsistency.

IMPORTANT: PARTIAL INPUTS. A solution may carry the reserved key "__partial__" set to true.
Such an input is a FRAGMENT of a solution -- for example a single route out of many -- offered
to show what one individual rule does and does not allow. When "__partial__" is present you must
judge ONLY what the fragment actually contains, and you must NOT report anything missing:
do not require the decisions to cover every entity, do not require an "objective_value", and
never label a fragment MALFORMED_SOLUTION for absent fields. Check every rule that can be
decided from the fragment alone, and ignore every rule that cannot. If nothing that is present
breaks a rule, set "feasible" to True and "status" to "PARTIAL_FEASIBLE". If something that is
present does break a rule, set "feasible" to False, set "status" to "PARTIAL_INFEASIBLE", and
say which rule. Return "objective_value" as None for a fragment. This key never appears on a
real candidate solution; it is used only while you are being checked.

When nothing is wrong, set "feasible" to True and "status" to "OK".

ALWAYS return your own recomputed objective in "objective_value" -- including in case (b), where it tells the caller what the solution actually achieves. Never copy the value from the input solution.

IMPORTANT: You MUST also RECOMPUTE the true objective value of the solution from the decisions themselves, independently of any "objective_value" field the solution reports, and return it under the key "objective_value" of your result dictionary. Return the recomputed number when the solution is feasible, and None when it is infeasible. This recomputed value is what will be used to compare two different solutions, so it must never be copied from the input solution.

IMPORTANT: BUILD THE OBJECTIVE ONE TERM AT A TIME AND ADD THE TERMS AT THE END. These objectives are nearly always a sum of separately defined quantities -- a setup cost, a holding cost, a travel cost, a penalty, a revenue that enters with a minus sign. Do not write the total as one expression. Give each term its own named variable, computed on its own, and combine them only in the last line:

    setup_cost = ...
    holding_cost = ...
    travel_cost = ...
    recomputed = setup_cost + holding_cost + travel_cost

Return those parts beside the total, under the key "objective_terms", as a dictionary from the name you gave each term to its value -- for example {{"setup_cost": 120.0, "holding_cost": 8.5, "travel_cost": 0.0}}. Include every term the description defines, including the ones that come out zero, and give each the sign the description gives it: a profit collected and a cost paid are two separate terms, not one netted number.

This is diagnostic, not cosmetic. A single wrong coefficient in one term rejects every correct solution, and a message saying only that the total differs gives the next attempt nothing to work with, while a breakdown says which part is wrong and by how much. For the same reason, when you reject a solution with status INCONSISTENT_OBJECTIVE, name the terms and their values in the message.

IMPORTANT: Your function MUST return a dictionary with the following format:
{{
    "feasible": bool,  # True if feasible, False otherwise
    "objective_value": float or None,  # objective RECOMPUTED from the decisions; None if infeasible
    "objective_terms": dict,  # the named parts you summed to reach
                   #   "objective_value", e.g. {{"setup_cost": 120.0,
                   #   "holding_cost": 8.5}}. Same names every call, zeros
                   #   included. Omit only when the objective truly has one term.
    "status": str,  # "OK", "MALFORMED_SOLUTION", "INFEASIBLE", "INCONSISTENT_OBJECTIVE",
                   #   or "PARTIAL_FEASIBLE"/"PARTIAL_INFEASIBLE" for a "__partial__" fragment
    "warnings": list,  # any other non-fatal notes
    "violated_constraints": list,  # List of constraint indices that are violated (e.g., [1, 3, 5])
    "violation_diagnose": list,  # One plain-language sentence per violation saying
                         # WHICH RULE OF THE REAL PROBLEM is broken, in the
                         # vocabulary of the problem description rather than of
                         # the model -- e.g. "the demand of the period is not
                         # met", "a vehicle leaves before its cargo is loaded".
                         # Not the index, not the arithmetic: the operating rule
                         # a planner would recognise. Same length and order as
                         # "violations".
    "violations": list,  # One self-contained sentence per violation. Each message
                         # MUST name: the rule that was broken; the indices where it
                         # was broken, using whatever index labels this problem uses;
                         # the value you computed; the limit or required value it had
                         # to satisfy; and the amount by which it misses. Write it so
                         # that a reader who cannot see this checker's source can
                         # locate and fix the error from the message alone.
                         # Shape: "Constraint <k> (<what the rule requires>), <indices>:
                         #         computed <value> against <required value>, off by <amount>"
                         # Not acceptable: "Constraint 3 violated." -- naming a rule
                         # without saying where it broke or by how much forces the
                         # reader to re-derive everything you already computed.
    "violation_magnitudes": list  # List of violation detail dictionaries
}}

Each violation magnitude should be a dict with these fields:
{{
    "constraint": int,  # The constraint index
    "lhs": float,  # Left-hand side value
    "rhs": float,  # Right-hand side value
    "raw_excess": float,  # Amount of violation
    "normalizer": float,  # Normalization factor
    "ratio": float  # Normalized ratio
}}

Use TOL = 1e-5 for numerical comparisons.

CODE FORMAT:
Return ONLY the complete Python code in a markdown code block.
"""
    code=get_response(prompt_simulator,model)
    return extract_code_block(code)

def _collections(obj, prefix="", depth=0, out=None):
    """Every collection in a JSON object, with its length and member shape."""
    if out is None:
        out = []
    if depth > 2:
        return out
    items = (obj.items() if isinstance(obj, dict)
             else enumerate(obj) if isinstance(obj, list) else [])
    for k, v in items:
        name = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, list) and v:
            m = v[0]
            kind = ("object" if isinstance(m, dict) else
                    "list" if isinstance(m, list) else type(m).__name__)
            out.append((name, len(v), "list of %s" % kind))
            if isinstance(m, dict):
                _collections(m, name + "[0]", depth + 1, out)
        elif isinstance(v, dict) and v:
            out.append((name, len(v), "dict keyed %r" % next(iter(v))))
            # Descend through ONE representative value, not all of them. A dict
            # keyed by scenario expands into one line per scenario otherwise, and
            # 250 of those bury the three lines that matter.
            k0 = next(iter(v))
            if isinstance(v[k0], (list, dict)) and v[k0]:
                _collections({k0: v[k0]}, name, depth + 1, out)
    return out


_KEYED = __import__("re").compile(r"^([A-Za-z][A-Za-z0-9]*)((?:_-?\d+)+)$")


def _key_shapes(solution, instance, max_fields=6):
    """Decompose composite dict keys -- 'x_0_0_1_1' -- into their components.

    A dict keyed by a plain index names its axis: bodur2017's x is keyed
    "0".."24" and a reader can see what it runs over. A dict keyed by a solver
    variable name packs three or four axes into one string and names none of
    them, so a reader has to guess which position is the item, which the
    machine, which the period. Guess wrong and every rule reads the wrong data
    at once.

    carvalho2022 is the case that forced this: told only ``production 215 (dict
    keyed 'x_0_0_1_1')``, six revisions in a row went looking for the decisions
    at the top level of the file and rejected a valid reference as malformed
    without ever saying which field was wrong.

    The ranges are in the keys and the sizes they should match are in the
    instance, so both are computed here instead of being left to inference.
    """
    if not isinstance(solution, dict):
        return []
    by_len = {}
    for name, n, _kind in _collections(instance):
        by_len.setdefault(n, []).append(name)

    # Sizes an axis could plausibly BE, by name: collection lengths, and the
    # numeric parameters a problem states about itself (num_warehouses, and the
    # like -- often as strings, so they are parsed rather than assumed).
    pool = [(name, n) for name, n, _k in _collections(instance)]
    if isinstance(instance, dict):
        for k, val in instance.items():
            try:
                iv = int(str(val))
            except (TypeError, ValueError):
                continue
            if 1 < iv < 100000:
                pool.append((k, iv))

    def _explain(n):
        """What an axis of size n is, when no single collection has that size.

        gruson2021's setup_variables index runs 0..87 and nothing in the
        instance has 88 of anything -- because the axis is every location at
        once: one plant, 20 warehouses, 67 retailers. Told only "88 distinct, no
        match", the build sized its array by the nearest number it could find,
        num_scenarios=100, and rejected the reference on every constraint at
        once for thirteen trials.

        Sums of two or three named sizes, optionally plus one for a depot or
        plant that has no collection of its own. Reported as a possibility and
        named, never asserted as the layout.
        """
        seen, best = set(), []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                for extra in ((), ) + tuple((k,) for k in range(len(pool))):
                    idx = (i, j) + extra
                    if len(set(idx)) != len(idx):
                        continue
                    tot = sum(pool[k][1] for k in idx)
                    for plus, label in ((0, ""), (1, " + 1")):
                        if tot + plus != n:
                            continue
                        key = tuple(sorted(pool[k][0] for k in idx)) + (plus,)
                        if key in seen:
                            continue
                        seen.add(key)
                        best.append(" + ".join("%s (%d)" % (pool[k][0], pool[k][1])
                                               for k in idx) + label)
            if len(best) >= 2:
                break
        return best[:2]

    out = []
    for field, v in solution.items():
        if not isinstance(v, dict) or not v or len(out) >= max_fields:
            continue
        parsed, stem = [], None
        for k in v:
            m = _KEYED.match(str(k))
            if not m:
                parsed = []
                break
            stem = stem or m.group(1)
            parsed.append([int(x) for x in m.group(2).lstrip("_").split("_")])
        if not parsed or len({len(p) for p in parsed}) != 1:
            continue
        width = len(parsed[0])
        letters = "abcdefgh"[:width]
        lines = ["    %-14s '%s_%s'  %d keys"
                 % (field, stem, "_".join("<%s>" % c for c in letters), len(v))]
        for i, c in enumerate(letters):
            vals = sorted({p[i] for p in parsed})
            span = vals[-1] + 1          # the axis is at least this long
            if len(vals) == 1:
                # A position that never varies is a constant the reference pins,
                # not an axis. Offering it a size invites an array with one
                # meaningless dimension.
                lines.append("        %s  always %d -- a fixed index, not an axis"
                             % (c, vals[0]))
                continue
            match = by_len.get(span) or by_len.get(len(vals)) or []
            if match:
                note = "   <- as many values as instance %s" % ", ".join(match[:3])
            else:
                alt = _explain(span)
                note = ("   <- nothing has %d of anything; it is %s"
                        % (span, " or ".join(alt))) if alt else \
                       "   <- no instance collection has %d entries" % span
            lines.append("        %s  %d..%d  %d distinct%s"
                         % (c, vals[0], vals[-1], len(vals), note))
        lines.append("      Only non-zero variables are stored, so a position "
                     "may skip values; size arrays from the instance, never "
                     "from how many distinct keys appear here.")
        out.append("\n".join(lines))
    return out


def _categories(instance, max_kinds=8):
    """Categorical fields inside instance collections, with how many of each.

    A rule almost always applies to one KIND of entity, and the kinds are in the
    data: nodes carry a type, arcs carry a class, items carry a category. The
    collection sizes alone do not show that, and a rule aimed at the wrong kind
    fails every reference at once.

    pecin2017 is the case in point. Its routes all end at node 201, its
    description says only "a single depot", and its Simulator applied "every
    customer appears on exactly one route" to node 201 -- reporting eleven
    duplicate visits and rejecting all twelve routes. The instance says plainly
    that node 0 is depot_source, node 201 is depot_sink and the other 200 are
    customers; nothing in the prompt had ever shown it.
    """
    out = []

    def walk(node, prefix, depth):
        if depth > 2 or len(out) >= 8:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, "%s.%s" % (prefix, k) if prefix else str(k), depth + 1)
        elif isinstance(node, list) and node and isinstance(node[0], dict):
            keys = [k for k, v in node[0].items() if isinstance(v, str)]
            for k in keys:
                vals = [e.get(k) for e in node if isinstance(e, dict)]
                distinct = {}
                for v in vals:
                    distinct[v] = distinct.get(v, 0) + 1
                    if len(distinct) > max_kinds:
                        break
                if 1 < len(distinct) <= max_kinds:
                    tally = ", ".join("%s=%d" % (a, b) for a, b
                                      in sorted(distinct.items(), key=lambda x: -x[1]))
                    out.append(("%s[].%s" % (prefix, k), tally))
    walk(instance, "", 0)
    return out


def _structure_map(instance, solution):
    """Which instance collection does each solution collection index into?

    The one thing a simulator cannot get from the description, and the one that
    breaks every rule at once when it is wrong. A solution entry like
    {"from": 12, "to": 662} names what it refers to and cannot be misaligned. A
    positional array -- scenarios[s]["e"][i] -- names nothing: the reader has to
    already know that s indexes the instance's scenarios and i its products.
    Guess an axis wrong and every constraint reads the wrong data, so the
    candidate fails every example identically and the failure surfaces as a
    constraint being wrong rather than an index.

    That is exactly what happened to bodur2017: it reported a capacity limit of
    0.0000 -- a facility that is not there -- and the revision, told the
    comparison was too strict, edited the comparison six times.

    Cardinality is computable, so it is computed and stated rather than left to
    be inferred. Sizes and field names only; no values, nothing about the rules.
    """
    inst_c = _collections(instance)
    sol_c = _collections(solution)
    if not sol_c:
        return ""

    def _indexable(kind):
        """Is this instance collection something a solution could INDEX INTO?

        A list of 33 customers is. A dict of six named fields is not -- its
        length is a field count, and offering it as an index target states
        something false. luo2017's depot is {id, x, y, ...}: six keys, so the
        map told the builder that a six-entry deliveries dict indexed the depot.
        It indexes customers by id. Six revisions later the build gave up.
        """
        if kind.startswith("list of"):
            return True
        if kind.startswith("dict keyed"):
            k = kind.split("dict keyed", 1)[1].strip().strip("'\"")
            return k.isdigit() or bool(_KEYED.match(k))
        return False

    by_len = {}
    for name, n, kind in inst_c:
        if _indexable(kind):
            by_len.setdefault(n, []).append(name)

    lines = ["DATA LAYOUT of this example, measured from the files themselves:",
             "  instance collections:"]
    for name, n, kind in inst_c[:14]:
        lines.append("    %-34s %d  (%s)" % (name, n, kind))
    cats = _categories(instance)
    if cats:
        lines.append("  what the entries of those collections ARE:")
        for name, tally in cats[:6]:
            lines.append("    %-34s %s" % (name, tally))
    lines.append("  solution collections:")
    for name, n, kind in sol_c[:14]:
        match = by_len.get(n) or []
        note = ("   <- same length as instance %s" % ", ".join(match[:3])) if match else ""
        lines.append("    %-34s %d  (%s)%s" % (name, n, kind, note))
    keyed = _key_shapes(solution, instance)
    if keyed:
        lines.append("  solution collections whose KEYS pack several indices into one "
                     "string, decomposed:")
        lines.extend(keyed)
        lines.append("  Read the decisions from inside these named fields, by taking "
                     "the key apart. They are not at the top level of the solution, "
                     "and there is no field per variable.")
    lines.append("")
    lines.append("A positional collection indexes the instance collection of the same "
                 "length. If your indices do not line up with these, the disagreement "
                 "is an index, not a comparison.")
    return "\n".join(lines)


def _abridge_for_prompt(obj, max_items=12, budget=60000):
    """Shrink a solution to its SHAPE before it goes into a prompt.

    The revision prompt pastes the failing example in whole. That is fine at
    12 kB and fatal at 50 MB: gangammanavar2020's reference solution is 49.6 MB
    and the request came back 413 (body over 16 MB), bodur2017's is 5.2 MB and
    came back 400 (input length over 995,904). In both cases the FIRST revision
    was unsendable, so the build died having never had a chance to correct
    itself -- and it looked like a modelling failure rather than a transport
    one.

    What the model needs from the example is its shape: which fields exist, how
    they nest, what the values look like. The ten-thousandth element of a list
    adds nothing. So keep every key, keep the first few members of every
    collection, and say how many were dropped -- then tighten and repeat until
    it fits.

    Structure is preserved exactly; only bulk is removed. A checker written
    against the abridged shape is the same checker.
    """
    def walk(o, n):
        if isinstance(o, dict):
            return {k: walk(v, n) for k, v in o.items()}
        if isinstance(o, list):
            head = [walk(v, n) for v in o[:n]]
            if len(o) > n:
                head.append("... %d more item(s) omitted" % (len(o) - n))
            return head
        return o

    if isinstance(obj, str):
        return obj if len(obj) <= budget else obj[:budget] + "\n... truncated"
    for n in (max_items, 6, 3, 1):
        try:
            text = json.dumps(walk(obj, n), indent=2, default=str)
        except Exception:
            return str(obj)[:budget]
        if len(text) <= budget:
            return text
    return text[:budget] + "\n... truncated"


def revise_simulator(description, code, inst, review, model, instance=None):
    if instance is not None:
        layout = _structure_map(instance, inst if isinstance(inst, dict) else {})
        if layout:
            review = "%s\n\n%s" % (review, layout)
    inst = _abridge_for_prompt(inst)
    prompt_simulator = f"""
You are an expert reviewer for operations research problem feasibility check.
You are generating a function for a specific OR problem to check the feasibility.
Problem description:
-----
{description}
-----
You have generated a code, but the code fails on a instance. Here are the previous code, failed instance and the review.
Previous code:
-----
{code}
-----
Failed instance:
-----
{inst}
-----
{review}

NOTE: the description provides necessary variables, your task is generating a function to check the feasibility. The input is a solution and the output is true or false. If true, it means that the solution is feasible
You should refer the failed instance to revise your code.
The form of your function should be: feasibility_check(data,solution)

IMPORTANT: If the solution contains an "objective_value" field, you MUST verify that the recomputed objective value matches the reported value. Include this as a constraint check (e.g., constraint index for objective validation) and report any mismatch as a violation with appropriate LHS/RHS values.

IMPORTANT: IMPORTANT: Distinguish three different reasons for rejecting a solution, and label them differently.

(a) MALFORMED_SOLUTION - the input does not express a decision in the syntax the problem description specifies: a required field is missing or misnamed, a collection that should hold the decisions is empty or is not a list, an entry is missing one of its required fields, or a value has the wrong type. You cannot judge such an input at all. Set "feasible" to False, set "status" to "MALFORMED_SOLUTION", say exactly which part of the required syntax is missing or wrong, and do NOT report any rule of the problem as violated -- nothing has been checked. Do this check FIRST, before any other, and be tolerant of extra fields the description does not mention.

(b) INFEASIBLE - the solution is well formed, but breaks one or more of the rules stated in the problem description above, so it could not be carried out. Set "feasible" to False, set "status" to "INFEASIBLE", and describe the rule that is violated.

(c) INCONSISTENT_OBJECTIVE - the solution is well formed and every rule stated in the problem description holds, so the solution could be carried out, but the reported "objective_value" differs from the value you recompute by more than 0.5. This is NOT a rule violation and must never be described as one. Set "feasible" to False so the solution is still rejected, set "status" to "INCONSISTENT_OBJECTIVE", and write a message saying the solution itself is valid while the reported objective is inconsistent, naming both numbers, for example: "Solution is valid, but the reported objective_value 9.0011 is inconsistent with the recomputed objective 3.0." Do NOT list any rule as violated in this case. A difference of 0.5 or less is expected, because the problem description may not fix the value of every constant appearing in the objective, and is not an inconsistency.

IMPORTANT: PARTIAL INPUTS. A solution may carry the reserved key "__partial__" set to true.
Such an input is a FRAGMENT of a solution -- for example a single route out of many -- offered
to show what one individual rule does and does not allow. When "__partial__" is present you must
judge ONLY what the fragment actually contains, and you must NOT report anything missing:
do not require the decisions to cover every entity, do not require an "objective_value", and
never label a fragment MALFORMED_SOLUTION for absent fields. Check every rule that can be
decided from the fragment alone, and ignore every rule that cannot. If nothing that is present
breaks a rule, set "feasible" to True and "status" to "PARTIAL_FEASIBLE". If something that is
present does break a rule, set "feasible" to False, set "status" to "PARTIAL_INFEASIBLE", and
say which rule. Return "objective_value" as None for a fragment. This key never appears on a
real candidate solution; it is used only while you are being checked.

When nothing is wrong, set "feasible" to True and "status" to "OK".

ALWAYS return your own recomputed objective in "objective_value" -- including in case (b), where it tells the caller what the solution actually achieves. Never copy the value from the input solution.

IMPORTANT: You MUST also RECOMPUTE the true objective value of the solution from the decisions themselves, independently of any "objective_value" field the solution reports, and return it under the key "objective_value" of your result dictionary. Return the recomputed number when the solution is feasible, and None when it is infeasible. This recomputed value is what will be used to compare two different solutions, so it must never be copied from the input solution.

IMPORTANT: BUILD THE OBJECTIVE ONE TERM AT A TIME AND ADD THE TERMS AT THE END. These objectives are nearly always a sum of separately defined quantities -- a setup cost, a holding cost, a travel cost, a penalty, a revenue that enters with a minus sign. Do not write the total as one expression. Give each term its own named variable, computed on its own, and combine them only in the last line:

    setup_cost = ...
    holding_cost = ...
    travel_cost = ...
    recomputed = setup_cost + holding_cost + travel_cost

Return those parts beside the total, under the key "objective_terms", as a dictionary from the name you gave each term to its value -- for example {{"setup_cost": 120.0, "holding_cost": 8.5, "travel_cost": 0.0}}. Include every term the description defines, including the ones that come out zero, and give each the sign the description gives it: a profit collected and a cost paid are two separate terms, not one netted number.

This is diagnostic, not cosmetic. A single wrong coefficient in one term rejects every correct solution, and a message saying only that the total differs gives the next attempt nothing to work with, while a breakdown says which part is wrong and by how much. For the same reason, when you reject a solution with status INCONSISTENT_OBJECTIVE, name the terms and their values in the message.

IMPORTANT: Your function MUST return a dictionary with the following format:
{{
    "feasible": bool,  # True if feasible, False otherwise
    "objective_value": float or None,  # objective RECOMPUTED from the decisions; None if infeasible
    "objective_terms": dict,  # the named parts you summed to reach
                   #   "objective_value", e.g. {{"setup_cost": 120.0,
                   #   "holding_cost": 8.5}}. Same names every call, zeros
                   #   included. Omit only when the objective truly has one term.
    "status": str,  # "OK", "MALFORMED_SOLUTION", "INFEASIBLE", "INCONSISTENT_OBJECTIVE",
                   #   or "PARTIAL_FEASIBLE"/"PARTIAL_INFEASIBLE" for a "__partial__" fragment
    "warnings": list,  # any other non-fatal notes
    "violated_constraints": list,  # List of constraint indices that are violated (e.g., [1, 3, 5])
    "violation_diagnose": list,  # One plain-language sentence per violation saying
                         # WHICH RULE OF THE REAL PROBLEM is broken, in the
                         # vocabulary of the problem description rather than of
                         # the model -- e.g. "the demand of the period is not
                         # met", "a vehicle leaves before its cargo is loaded".
                         # Not the index, not the arithmetic: the operating rule
                         # a planner would recognise. Same length and order as
                         # "violations".
    "violations": list,  # One self-contained sentence per violation. Each message
                         # MUST name: the rule that was broken; the indices where it
                         # was broken, using whatever index labels this problem uses;
                         # the value you computed; the limit or required value it had
                         # to satisfy; and the amount by which it misses. Write it so
                         # that a reader who cannot see this checker's source can
                         # locate and fix the error from the message alone.
                         # Shape: "Constraint <k> (<what the rule requires>), <indices>:
                         #         computed <value> against <required value>, off by <amount>"
                         # Not acceptable: "Constraint 3 violated." -- naming a rule
                         # without saying where it broke or by how much forces the
                         # reader to re-derive everything you already computed.
    "violation_magnitudes": list  # List of violation detail dictionaries
}}

Each violation magnitude should be a dict with these fields:
{{
    "constraint": int,  # The constraint index
    "lhs": float,  # Left-hand side value
    "rhs": float,  # Right-hand side value
    "raw_excess": float,  # Amount of violation
    "normalizer": float,  # Normalization factor
    "ratio": float  # Normalized ratio
}}

Use TOL = 1e-5 for numerical comparisons.

CODE FORMAT:
Return ONLY the complete Python code in a markdown code block.
"""
    code=get_response(prompt_simulator,model)
    return extract_code_block(code)

def fix_simulator_syntax(desc, code, error_msg, model):
    prompt = f"""
You are an expert Python developer. The simulator code has syntax errors.

PROBLEM DESCRIPTION:
{desc}

ERROR MESSAGE:
{error_msg}

CODE WITH ERRORS:
```python
{code}
Fix ALL syntax errors. Make sure:

Function signature: feasibility_check(data, solution)

Return format:
{{
"feasible": bool,
"objective_value": float or None,
"status": str,
"warnings": list,
"violated_constraints": list,
"violations": list,
"violation_magnitudes": list
}}

All imports are included

All variables are defined before use

All parentheses and quotes are properly closed

Return ONLY the complete fixed Python code in a markdown code block.
"""
    response = get_response(prompt, model)
    return extract_code_block(response)


# ---------------------------------------------------------------------------
# Reusable builder: OSCAR rebuilds and re-certifies its own Simulator on every
# run, so this has to be callable, write to a run-local path, and never touch
# the module-level ``simulator`` import.
# ---------------------------------------------------------------------------

def _load_module(path):
    import importlib.util
    name = "sim_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_OBJ_BOOKKEEPING = {
    "objective_value", "runtime", "status", "solver", "mip_gap", "best_bound",
    "solve_time", "node_count", "status_name", "status_str", "instance_id",
    "gap", "solver_status", "optimality_gap", "runtime_s", "num_variables",
    "num_constraints", "seed", "seed_used", "generation_attempt",
}


_PRICED = ("cost", "price", "holding", "setup", "penalty", "profit", "revenue",
           "fee", "charge", "tariff", "rate")


def _priced_and_decided(instance, reference_solution):
    """What the instance puts a PRICE on, and what the solution DECIDES.

    A term goes missing from an objective silently. gruson2021 prices holding at
    three levels -- plant, warehouse, retailer -- and its reference reports an
    inventory collection for each; the candidate charged one of them, returned a
    single "holding_cost" term, and came out 104,483 short with every structural
    rule passing. Nothing in the failure said a level was missing, so nine
    revisions adjusted the one term it had.

    Both halves are readable from the files, so both are stated and the reader
    is left to pair them.
    """
    priced, decided = [], []
    if isinstance(instance, dict):
        for k, v in instance.items():
            if not any(w in k.lower() for w in _PRICED):
                continue
            if isinstance(v, (list, tuple)) and v:
                priced.append("%s (%d values)" % (k, len(v)))
            elif isinstance(v, dict) and v:
                priced.append("%s (%d entries)" % (k, len(v)))
            else:
                try:
                    priced.append("%s (%s)" % (k, float(str(v))))
                except (TypeError, ValueError):
                    continue
    if isinstance(reference_solution, dict):
        for k, v in reference_solution.items():
            if k in _OBJ_BOOKKEEPING or not isinstance(v, (list, dict)) or not v:
                continue
            decided.append("%s (%d)" % (k, len(v)))
    return priced[:14], decided[:14]


def _objective_parts(candidate_result, reference_solution, instance=None):
    """What both sides say the objective is MADE OF, for a revision to compare.

    An objective is a sum of separately defined quantities, and a candidate
    whose total is wrong is wrong in one of them. "Your objective differs" sends
    the next attempt to re-read the whole objective; "your travel cost is the
    part that differs" sends it to one line.

    Two sources, both already to hand. The candidate now returns the terms it
    summed. And a reference often reports its own components beside the total --
    colombi2017 states total_profit and total_travel_cost next to an objective
    that is their difference -- which says how many terms there are and what
    each should come to, without anyone having to guess.
    """
    out = []
    terms = (candidate_result or {}).get("objective_terms")
    if isinstance(terms, dict) and terms:
        out.append(" You summed these terms to reach your objective: %s."
                   % ", ".join("%s=%s" % (k, v) for k, v in list(terms.items())[:12]))
    else:
        out.append(" You did not return \"objective_terms\", so there is no way "
                   "to see which part of the objective is wrong. Compute each "
                   "term of the objective into its own named variable, add them "
                   "at the end, and return them under \"objective_terms\".")
    comps = {}
    if isinstance(reference_solution, dict):
        for k, v in reference_solution.items():
            if k in _OBJ_BOOKKEEPING or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                comps[k] = v
    if comps:
        out.append(" The solution itself reports these quantities alongside its "
                   "objective: %s. Some or all of them are the parts the "
                   "objective is assembled from, so line your terms up against "
                   "them and find the one that disagrees -- but recompute each "
                   "from the decisions, never copy these numbers."
                   % ", ".join("%s=%s" % (k, v) for k, v in list(comps.items())[:12]))

    priced, decided = _priced_and_decided(instance, reference_solution)
    if priced:
        n_terms = len(terms) if isinstance(terms, dict) else 0
        out.append(" The instance puts a price on: %s." % ", ".join(priced))
        if decided:
            out.append(" The solution decides: %s." % ", ".join(decided))
        out.append(" You returned %d objective term(s). Every priced quantity "
                   "belongs to a term unless the description says otherwise, "
                   "and a quantity priced at several levels -- one cost per "
                   "level, one decision collection per level -- needs a term "
                   "for EACH level, not one. Before changing a term you already "
                   "have, check which priced quantity your objective never "
                   "touches." % n_terms)
    return "".join(out)


def _check_examples(fn, examples, base_dir, obj_tol=0.2, verbose=True):
    """Run the candidate simulator over the certification set.

    Returns (ok, failed_example, review_text). A candidate must get every
    feasibility verdict right AND return the recorded objective value for the
    feasible examples, since certification compares objectives.
    """
    for n, ex in enumerate(examples, 1):
        with open(os.path.join(base_dir, ex["instance"]), "r", encoding="utf-8") as f:
            data = json.load(f)
        partial = bool(ex.get("partial"))
        if partial:
            # a fragment supplied inline, showing what one rule does or does not allow
            sol = dict(ex["solution"])
            sol["__partial__"] = True
            name = "partial: %s" % ex.get("rule", "")
        else:
            with open(os.path.join(base_dir, ex["solution"]), "r", encoding="utf-8") as f:
                sol = json.load(f)
            name = ex["solution"]
        want = ex["feasible"]

        def report(ok, got_txt, why=""):
            if verbose:
                print("    %d/%d %-36s want=%-5s got=%-14s %s%s"
                      % (n, len(examples), name[:36], want, got_txt,
                         "OK" if ok else "FAIL", (" - " + why) if why else ""),
                      flush=True)

        try:
            res = fn(data, sol)
        except Exception as e:
            report(False, "exception", "%s: %s" % (type(e).__name__, str(e)[:50]))
            return False, ex, ("Your function raised %s on this example: %s. "
                               "It must return a result dictionary for every input."
                               % (type(e).__name__, e)), e
        if not isinstance(res, dict):
            # A candidate that falls off the end of a branch returns None.  That
            # is a fixable slip, not a reason to abandon the build, so hand it
            # back as a revision instead of crashing on res.get(...).
            report(False, "returned %s" % type(res).__name__)
            return False, ex, (
                "Your function returned %r instead of a result dictionary. It must "
                "return a dict carrying at least \"feasible\", \"status\" and "
                "\"violations\" on EVERY path through the function, including the "
                "paths that currently fall off the end without a return statement."
                % (res,)), None
        got = res.get("feasible")
        if partial:
            # A fragment is judged on what it contains: the verdict plus a status that
            # says the simulator understood it was looking at a fragment.
            expect = "PARTIAL_FEASIBLE" if want else "PARTIAL_INFEASIBLE"
            got_status = str(res.get("status") or "").upper()
            if got != want or got_status != expect:
                report(False, "%s/%s" % (got, got_status or "no status"), "want %s/%s" % (want, expect))
                return False, ex, (
                    'This input is a FRAGMENT of a solution: it carries "__partial__": true and '
                    "shows only what one rule does or does not allow. %s Judge only what the "
                    "fragment contains -- never call it malformed for anything absent, and do not "
                    'require an "objective_value". Here you must return feasible=%s with status '
                    '"%s", and you returned feasible=%s with status "%s".'
                    % (ex.get("reason") or "", want, expect, got, got_status or "nothing")), None
            report(True, "%s status=%s" % (got, got_status))
            continue
        if got == want and not want and ex.get("expect_status"):
            got_status = str(res.get("status") or "").upper()
            if got_status != ex["expect_status"]:
                report(False, "status=%s" % (got_status or "missing"),
                       "want %s" % ex["expect_status"])
                return False, ex, (
                    "You correctly rejected this example, but its status must be '%s' and you "
                    "returned '%s'. %s Set the \"status\" field accordingly and make the message "
                    "match it." % (ex["expect_status"], got_status or "nothing",
                                   ex.get("reason") or "")), None
        if got != want:
            report(False, str(got))
            if want:
                # Hand back the candidate's OWN violation messages. Without them the
                # revision prompt says only "something wrongly fires", which is not
                # enough to locate an off-by-one in a specific check.
                said = (res.get("violations") or [])[:5]
                detail = ("" if not said else
                          " Your function reported these violations, and every one of them is "
                          "wrong for this solution: " + " | ".join(str(v) for v in said))
                if not said:
                    # No violated constraint at all: the rejection came from somewhere
                    # other than a constraint check -- a malformed-input guard, or an
                    # objective the candidate recomputed and disagreed with.  Hand back
                    # what the candidate itself said, or the revision has nothing to
                    # work from and will go looking for a too-strict comparison that
                    # does not exist.
                    detail = (" Your function reported NO violated constraint, so the "
                              "rejection did not come from a constraint check. It returned "
                              "status %r%s. Fix whatever produced that: in particular, if the "
                              "objective you recompute disagrees with the reported one, it is "
                              "your reading of the input data that is wrong, not the solution."
                              % (str(res.get("status") or "none"),
                                 (" and warnings: " + " | ".join(
                                     str(w) for w in (res.get("warnings") or [])[:5]))
                                 if res.get("warnings") else ""))
                    if str(res.get("status") or "").upper() == "INCONSISTENT_OBJECTIVE":
                        detail += _objective_parts(res, sol, data)
                # One kind of mistake CAN be named, because the example itself
                # settles it. A recorded reference is by construction written in
                # the required syntax, so a MALFORMED verdict on it is never a
                # judgement about the solution -- the code did not manage to read
                # the file. Left to the generic text below, six revisions in a row
                # went looking for a rule that was too strict while the decisions
                # sat unread inside a field the parser never opened.
                if str(res.get("status") or "").upper() == "MALFORMED_SOLUTION":
                    return False, ex, (
                        "You rejected this example as MALFORMED_SOLUTION, but it is a "
                        "recorded reference solution: it IS written in the syntax the "
                        "problem requires. Neither the rules nor the objective are in "
                        "question here -- your code did not manage to READ the file. "
                        + ("It did not even say which field was wrong, which is itself a "
                           "fault to fix: a MALFORMED_SOLUTION verdict must name the field "
                           "and say what is wrong with it. " if not said else
                           "It reported: " + " | ".join(str(v) for v in said) + ". ")
                        + "Do not change any constraint and do not change the objective. "
                          "Find the decisions where they actually are. The layout below "
                          "gives the field names, the size of each collection, and -- "
                          "where the keys pack several indices into one string -- what "
                          "each position of the key runs over."), None
                # Do not assert WHICH kind of mistake it is. Telling the model the
                # comparison is too strict sends it to relax a threshold, and when
                # the fault is an index -- reading scenario s's data for scenario
                # s+1, or a facility that is not there -- that is the wrong repair
                # and it will be made again on the next revision, and the next.
                return False, ex, ("This example is recorded as FEASIBLE, but your function "
                                   "returned infeasible." + detail +
                                   " There are two different mistakes that produce this, and the "
                                   "numbers above tell you which. If a limit or required value you "
                                   "report is zero, empty, or otherwise not what that quantity "
                                   "should be for this data, you are READING THE WRONG PART OF THE "
                                   "INPUT -- an index, a key, or an axis is misaligned, and the "
                                   "layout below says what aligns with what. Only if the values "
                                   "you report are the right ones is the comparison itself too "
                                   "strict; in that case re-read the description for the rule "
                                   "behind each check."
                                   + ((" Note: " + ex["note"]) if ex.get("note") else "")), None
            return False, ex, ("This example is recorded as INFEASIBLE, but your function returned "
                               "feasible. The recorded reason is: " + (ex.get("reason") or "") +
                               " Your function must detect this and report it as a violation. Do not "
                               "silently repair or override values reported in the solution; check "
                               "them as given."), None
        if "objective_value" not in res:
            report(False, str(got), "no objective_value key")
            return False, ex, ('Your result dictionary has no "objective_value" key. It must '
                               "always be present: the recomputed objective when the solution is "
                               "feasible, and None when it is infeasible."), None
        got_obj = res.get("objective_value")
        if want and ex.get("objective_value") is not None:
            if got_obj is None or abs(float(got_obj) - float(ex["objective_value"])) > obj_tol:
                report(False, "obj=%s" % got_obj, "want obj=%s" % ex["objective_value"])
                return False, ex, ("This example is feasible with a recorded objective value of %s, "
                                   "but your function returned %s. Recompute the objective from the "
                                   'decisions themselves and return it under "objective_value".'
                                   % (ex["objective_value"], got_obj)
                                   + _objective_parts(res, sol, data)
                                   + " The difference is %s, which is the size of whatever single "
                                     "term is wrong -- look for a term that would be off by that "
                                     "much before rewriting the whole objective."
                                   % (None if got_obj is None else
                                      round(float(got_obj) - float(ex["objective_value"]), 6))), None
        report(True, "%s obj=%s" % (got, got_obj))
    return True, None, "", None


def _run_battery(fn, base_dir, verbose=True):
    """Generated malformed/tolerance inputs. Returns (ok, failures)."""
    try:
        import json as _json
        import problem
        import robustness
        ex = _json.load(open(os.path.join(base_dir, "certification_examples.json"),
                             encoding="utf-8"))
        pos = next(e for e in ex if e["feasible"])
        inst = _json.load(open(os.path.join(base_dir, pos["instance"]), encoding="utf-8"))
        good = _json.load(open(os.path.join(base_dir, pos["solution"]), encoding="utf-8"))
        keep = set(problem.SOLUTION_FIELDS) | set(problem.SOLUTION_SCALAR_FIELDS) \
               | {"objective_value"}
        if problem.SOLUTION_LIST_FIELD:
            keep.add(problem.SOLUTION_LIST_FIELD)
        keep.discard(None)
        good = {k: v for k, v in good.items() if k in keep}
        return robustness.run_battery(fn, inst, good, pos.get("objective_value"),
                                      verbose=verbose)
    except Exception as e:
        if verbose:
            print("    battery could not run: %s" % e)
        return True, []          # never block certification on a harness fault


def _battery_review(failures):
    lines = ["Your function was given inputs that are not valid solutions, and mishandled "
             "some of them. It must NEVER raise an exception, and it must reject any input "
             "that does not express a decision in the required syntax with "
             "\"status\": \"MALFORMED_SOLUTION\" and \"feasible\": false. Inputs it must also "
             "accept unchanged are marked below as tolerance cases.",
             "",
             "Cases handled incorrectly:"]
    for name, kind, what in failures[:12]:
        lines.append("  - [%s] %s -> %s" % (kind, name, what))
    lines.append("")
    lines.append("Validate the shape of the input BEFORE using any of its parts: check that it "
                 "is an object, that every required field is present, that the collection of "
                 "decisions is a non-empty list, that every entry is an object carrying each "
                 "required field with a sensible type, and that required scalar fields are "
                 "numbers. Return the malformed verdict at the first problem found. Extra "
                 "fields the description does not mention must be ignored, not rejected.")
    return chr(10).join(lines)


def build_certified_simulator(desc, examples, model, out_path, base_dir=None,
                              max_trials=8, max_syntax_fix=6, verbose=True,
                              stuck_after=3):
    """Build a Simulator oracle and certify it against ``examples``.

    Prints progress at every step: a single LLM call here can take minutes, and
    without output there is no way to tell a slow draft from a loop that will
    never converge. Bails out when the same example keeps failing the same way.

    Returns (module, log). ``log["certified"]`` says whether every example passed.
    """
    base_dir = base_dir or os.path.dirname(os.path.abspath(out_path))
    log = {"trials": 0, "syntax_fixes": 0, "revisions": 0, "certified": False,
           "failures": [], "seconds": 0.0}
    t_start = time.time()

    def say(msg):
        if verbose:
            print("[sim %5.0fs] %s" % (time.time() - t_start, msg), flush=True)

    say("drafting simulator with %s ..." % model)
    t = time.time()
    with call_role("prep_simulator_generate"):
        code = generate_simulator(desc, model)
    say("draft returned after %.0fs (%d chars)" % (time.time() - t, len(code)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)

    seen = []
    # Highest-scoring candidate so far, and its score. A revision that
    # regresses would otherwise become the base for every later trial.
    best_code, best_score = code, -1
    for trial in range(max_trials):
        log["trials"] = trial + 1
        say("trial %d/%d: importing" % (trial + 1, max_trials))
        mod = None
        for fix in range(max_syntax_fix):
            try:
                mod = _load_module(out_path)
                if not hasattr(mod, "feasibility_check"):
                    raise AttributeError("no feasibility_check(data, solution) defined")
                break
            except Exception as e:
                log["syntax_fixes"] += 1
                say("  will not import (%s); syntax fix %d/%d"
                    % (str(e)[:70], fix + 1, max_syntax_fix))
                t = time.time()
                with call_role("prep_simulator_syntaxfix"):
                    code = fix_simulator_syntax(desc, code, str(e), model)
                say("  syntax fix returned after %.0fs" % (time.time() - t))
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                mod = None
        if mod is None:
            msg = "could not import the simulator after %d syntax fixes" % max_syntax_fix
            say("GIVING UP: " + msg)
            log["failures"].append(msg)
            break

        say("  certifying against %d example(s)" % len(examples))
        ok, failed, review, exc = _check_examples(mod.feasibility_check, examples, base_dir,
                                                  verbose=verbose)
        if ok:
            # Hand-written examples only test the shapes someone thought of. The
            # battery generates every way the schema allows an input to be wrong,
            # which is how a branch that no example reaches gets exercised.
            say("  robustness battery (generated malformed inputs)")
            b_ok, b_fails = _run_battery(mod.feasibility_check, base_dir, verbose)
            if not b_ok:
                ok = False
                failed = {"solution": "robustness battery", "reason": "", "synthetic": True}
                review = _battery_review(b_fails)
                exc = None
        if ok:
            log["certified"] = True
            log["seconds"] = round(time.time() - t_start, 1)
            say("CERTIFIED after %d trial(s), %d syntax fix(es), %d revision(s), %.0fs"
                % (log["trials"], log["syntax_fixes"], log["revisions"], log["seconds"]))
            return mod, log

        # how far this candidate got, so a regression can be detected
        if failed.get("synthetic"):
            score = len(examples) + 1          # cleared every example
        else:
            try:
                score = next(i for i, e in enumerate(examples)
                             if e.get("solution") == failed.get("solution"))
            except StopIteration:
                score = 0
        if score > best_score:
            best_score, best_code = score, code
        elif score < best_score:
            say("  regressed (reached %d, best was %d) -> revising from the best "
                "candidate instead" % (score, best_score))
            code = best_code

        sig = (failed["solution"], review[:80])
        log["failures"].append({"solution": failed["solution"], "review": review[:200]})
        if seen.count(sig) >= max(1, stuck_after - 1):
            msg = ("stuck: %s failed the same way %d times; revision is not "
                   "converging" % (failed["solution"], stuck_after))
            say("GIVING UP: " + msg)
            log["failures"].append(msg)
            break
        seen.append(sig)

        # A bare coding slip goes to the cheap targeted fixer; only a genuine
        # disagreement about the problem is worth a full revision.
        CODE_SLIPS = (NameError, AttributeError, TypeError, IndexError, KeyError,
                      UnboundLocalError)
        t = time.time()
        if exc is not None and isinstance(exc, CODE_SLIPS):
            say("  %s is a coding slip -> cheap syntax fix (trial %d/%d)"
                % (type(exc).__name__, trial + 1, max_trials))
            with call_role("prep_simulator_syntaxfix"):
                code = fix_simulator_syntax(desc, code, review, model)
            log["syntax_fixes"] += 1
            say("  syntax fix returned after %.0fs" % (time.time() - t))
        else:
            say("  revising (trial %d/%d)" % (trial + 1, max_trials))
            if failed.get("partial"):
                # a fragment is carried inline, not stored as a file
                inst_text = dict(failed["solution"])
                inst_text["__partial__"] = True
            elif failed.get("synthetic"):
                # generated inputs have no file behind them; the review carries the detail
                inst_text = failed.get("solution")
            else:
                with open(os.path.join(base_dir, failed["solution"]), "r", encoding="utf-8") as f:
                    inst_text = json.load(f)
            # The instance that goes with the failing solution, so the revision can
            # be told which instance collection each solution collection indexes.
            failed_instance = None
            try:
                with open(os.path.join(base_dir, failed["instance"]), "r",
                          encoding="utf-8") as f:
                    failed_instance = json.load(f)
            except Exception:
                pass
            with call_role("prep_simulator_revise"):
                code = revise_simulator(desc, code, inst_text, review, model,
                                        instance=failed_instance)
            log["revisions"] += 1
            say("  revision returned after %.0fs" % (time.time() - t))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)

    log["seconds"] = round(time.time() - t_start, 1)
    log["best_examples_passed"] = best_score
    say("NOT CERTIFIED after %.0fs (best candidate cleared %d example(s))"
        % (log["seconds"], best_score))
    return None, log


if __name__ == "__main__":
    model = os.environ.get("OSCAR_SMALL_MODEL", "qwen3.5-flash")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, "desc.txt"), "r", encoding="utf-8") as f:
        desc = f.read()
    with open(os.path.join(BASE_DIR, "certification_examples.json"), "r", encoding="utf-8") as f:
        examples = json.load(f)
    out_path = os.path.join(BASE_DIR, "simulator.py")
    mod, log = build_certified_simulator(desc, examples, model, out_path, base_dir=BASE_DIR)
    print(json.dumps(log, indent=2, default=str))
