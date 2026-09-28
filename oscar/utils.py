import contextlib
import os
import random
import time
import json
try:
    from groq import Groq
except ImportError:  # only needed for the llama3-70b path, unused here
    Groq = None
import openai

groq_key = os.environ.get("GROQ_API_KEY", "")
#openai_org = "https://api.deepseek.com"
openai_org = os.environ.get("OSCAR_API_BASE", "")
openai_key = os.environ.get("OSCAR_API_KEY", "")

groq_client = Groq(api_key=groq_key) if (Groq is not None and groq_key) else None
class _LazyClient(object):
    """An OpenAI client built on first use, so importing needs no credentials."""

    def __init__(self, key_env, base_env):
        self._key_env, self._base_env, self._client = key_env, base_env, None

    def __getattr__(self, name):
        if self._client is None:
            key = os.environ.get(self._key_env, "")
            if not key:
                raise RuntimeError(
                    "%s is not set, so no model can be called. See README.md."
                    % self._key_env)
            self._client = openai.OpenAI(
                api_key=key, base_url=os.environ.get(self._base_env) or None)
        return getattr(self._client, name)


open_ai_client = _LazyClient("OSCAR_API_KEY", "OSCAR_API_BASE")


def extract_json_from_end(text):
    
    try:
        return extract_json_from_end_backup(text)
    except:
        pass
    
    # Find the start of the JSON object
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found in the text.")

    # Extract text starting from the first '{'
    json_text = text[json_start:]
    
    # Remove backslashes used for escaping in LaTeX or other formats
    json_text = json_text.replace("\\", "")

    # Remove any extraneous text after the JSON end
    ind = len(json_text) - 1
    while json_text[ind] != "}":
        ind -= 1
    json_text = json_text[: ind + 1]

    # Find the opening curly brace that matches the closing brace
    ind -= 1
    cnt = 1
    while cnt > 0 and ind >= 0:
        if json_text[ind] == "}":
            cnt += 1
        elif json_text[ind] == "{":
            cnt -= 1
        ind -= 1

    # Extract the JSON portion and load it
    json_text = json_text[ind + 1:]

    # Attempt to load JSON
    try:
        jj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

    return jj

def extract_json_from_end_backup(text):

    if "```json" in text:
        text = text.split("```json")[1]
        text = text.split("```")[0]
    ind = len(text) - 1
    while text[ind] != "}":
        ind -= 1
    text = text[: ind + 1]

    ind -= 1
    cnt = 1
    while cnt > 0:
        if text[ind] == "}":
            cnt += 1
        elif text[ind] == "{":
            cnt -= 1
        ind -= 1

    # find comments in the json string (texts between "//" and "\n") and remove them
    while True:
        ind_comment = text.find("//")
        if ind_comment == -1:
            break
        ind_end = text.find("\n", ind_comment)
        text = text[:ind_comment] + text[ind_end + 1 :]

    # convert to json format
    jj = json.loads(text[ind + 1 :])
    return jj

# Which stage of the pipeline is making the current call. Single process, single
# thread, so a module global is enough; every metrics record carries it, which is
# what lets cost be broken down by call type rather than only totalled.
_CALL_ROLE = {"role": None, "tag": None}


MAX_API_RETRIES = int(os.environ.get("OSCAR_API_RETRIES", 5))


def _with_retries(fn):
    """Call fn(), retrying transient API failures with exponential backoff.

    Under 20-way concurrency, rate limits and gateway hiccups are routine; without
    this a single 429 loses a whole attempt and biases the results toward failure.
    Returns (result, retries_taken).
    """
    last = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return fn(), attempt
        except Exception as e:
            last = e
            if attempt == MAX_API_RETRIES - 1:
                break
            time.sleep(min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.5))
    raise last


def set_call_role(role=None, tag=None):
    _CALL_ROLE["role"] = role
    _CALL_ROLE["tag"] = tag


@contextlib.contextmanager
def call_role(role, tag=None):
    prev = dict(_CALL_ROLE)
    set_call_role(role, tag)
    try:
        yield
    finally:
        _CALL_ROLE.update(prev)


def append_jsonl(path, record):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def summarize_metrics(path):
    summary = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_elapsed_sec": 0.0,
    }
    if not path or not os.path.exists(path):
        return summary
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            summary["calls"] += 1
            summary["prompt_tokens"] += rec.get("prompt_tokens") or 0
            summary["completion_tokens"] += rec.get("completion_tokens") or 0
            summary["total_tokens"] += rec.get("total_tokens") or 0
            summary["llm_elapsed_sec"] += rec.get("elapsed_sec") or 0.0
    summary["llm_elapsed_sec"] = round(summary["llm_elapsed_sec"], 3)
    return summary

_claude_client = None


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic
        _claude_client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    return _claude_client


def _claude_call(prompt, model):
    """One call to a Claude model. Returns (text, usage)."""
    client = _get_claude_client()
    # The Qwen path runs with reasoning_effort="high" and thinking enabled, so a
    # Claude call without it is not a matched comparison -- it answers with no
    # thinking budget while its competitors get a large one.  Adaptive is the
    # current form for Opus 5 / Fable 5; budget_tokens is rejected on them.
    with client.messages.stream(
        model=model,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude refused the request: %r"
                           % getattr(msg, "stop_details", None))
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, msg.usage


def get_response_claude(prompt, model):
    started = time.perf_counter()
    (res, usage), retries = _with_retries(lambda: _claude_call(prompt, model))
    elapsed = time.perf_counter() - started
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "input_tokens", None),
        "completion_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": (getattr(usage, "input_tokens", 0) or 0)
                        + (getattr(usage, "output_tokens", 0) or 0),
        # cache reads are billed well below fresh input, so a run's real cost
        # cannot be recovered from input_tokens alone
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "prompt_chars": len(prompt),
        "response_chars": len(res) if res else 0,
        "role": _CALL_ROLE["role"],
        "tag": _CALL_ROLE["tag"],
        "retries": retries,
    }
    append_jsonl(os.environ.get("LLM_METRICS_PATH"), record)
    return res


# "llama3-70b-8192"
def get_response(prompt, model="llama3-70b-8192"):
    if model.startswith("claude"):
        return get_response_claude(prompt, model)
    if model == "llama3-70b-8192":
        client = groq_client
    elif model == "deepseek-v4-flash" or model == "deepseek-v4-pro":
        client = openai.OpenAI(api_key=os.environ.get("OSCAR_API_KEY", ""),base_url="https://api.deepseek.com")
    else:
        client = open_ai_client
    started = time.perf_counter()
    chat_completion, retries = _with_retries(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    ))
    elapsed = time.perf_counter() - started
    res = chat_completion.choices[0].message.content
    usage = getattr(chat_completion, "usage", None)
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        # reasoning models bill their thinking inside completion_tokens; itemise
        # it so a run's reasoning share can be separated after the fact
        "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None),
                                    "reasoning_tokens", None) if usage else None,
        "cached_prompt_tokens": getattr(getattr(usage, "prompt_tokens_details", None),
                                        "cached_tokens", None) if usage else None,
        "prompt_chars": len(prompt),
        "response_chars": len(res) if res else 0,
        "role": _CALL_ROLE["role"],
        "tag": _CALL_ROLE["tag"],
        "retries": retries,
    }
    append_jsonl(os.environ.get("LLM_METRICS_PATH"), record)

    return res

def get_response_revise(prompt):
    client_revise = openai.OpenAI(api_key=os.environ.get("OSCAR_API_KEY", ""), base_url=os.environ.get("OSCAR_API_BASE", ""))
    model="qwen3-coder-flash"
    started = time.perf_counter()
    chat_completion, retries = _with_retries(lambda: client_revise.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    ))
    elapsed = time.perf_counter() - started
    res = chat_completion.choices[0].message.content
    usage = getattr(chat_completion, "usage", None)
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        # reasoning models bill their thinking inside completion_tokens; itemise
        # it so a run's reasoning share can be separated after the fact
        "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None),
                                    "reasoning_tokens", None) if usage else None,
        "cached_prompt_tokens": getattr(getattr(usage, "prompt_tokens_details", None),
                                        "cached_tokens", None) if usage else None,
        "prompt_chars": len(prompt),
        "response_chars": len(res) if res else 0,
        "role": _CALL_ROLE["role"],
        "tag": _CALL_ROLE["tag"],
        "retries": retries,
    }
    append_jsonl(os.environ.get("LLM_METRICS_PATH"), record)

    return res

if __name__ == "__main__":
    
    text = 'To maximize the number of successfully transmitted shows, we can introduce a new variable called "TotalTransmittedShows". This variable represents the total number of shows that are successfully transmitted.\n\nThe constraint can be formulated as follows:\n\n\\[\n\\text{{Maximize }} TotalTransmittedShows\n\\]\n\nTo model this constraint in the MILP formulation, we need to add the following to the variables list:\n\n\\{\n    "TotalTransmittedShows": \\{\n        "shape": [],\n        "type": "integer",\n        "definition": "The total number of shows transmitted"\n    \\}\n\\}\n\nAnd the following auxiliary constraint:\n\n\\[\n\\forall i \\in \\text{{NumberOfShows}}, \\sum_{j=1}^{\\text{{NumberOfStations}}} \\text{{Transmitted}}[i][j] = \\text{{TotalTransmittedShows}}\n\\]\n\nThe complete output in the requested JSON format is:\n\n\\{\n    "FORMULATION": "",\n    "NEW VARIABLES": \\{\n        "TotalTransmittedShows": \\{\n            "shape": [],\n            "type": "integer",\n            "definition": "The total number of shows transmitted"\n        \\}\n    \\},\n    "AUXILIARY CONSTRAINTS": [\n        ""\n    ]\n\\'
    
    extract_json_from_end(text)