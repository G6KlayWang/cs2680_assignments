"""Course API settings (endpoint, API key, model ids) and the harness tunables.

The course API settings are environment variables: CS2680_API_KEY and the model ids
CS2680_MODEL_EXPERT / CS2680_MODEL_STANDARD / CS2680_MODEL_STARTER are exported on the host
and passed into the agent container by run_all.py (every CS2680_* variable is forwarded);
CS2680_BASE_URL is set by madsOpt.py. The helpers below read them with sensible fallbacks.

Every tunable can be overridden with an environment variable for local runs. Inside the
leaderboard's agent container only MADSOPT_MAX_ITERATIONS is forwarded, so the defaults
written here are what the leaderboard runs with.
"""

import os
import sys

_WARNED_MODEL_VARS: set = set()

# >>> CS2680 API KEY MACRO <<<
# Fill this in for local development if you don't want to export the env var.
# The CS2680_API_KEY environment variable always takes precedence, and this
# constant must stay empty ("") in the submitted repo — never commit a key.
CS2680_API_KEY = ""

CS2680_BASE_URL = "https://api.cs2680.com/v1"   # fallback; madsOpt.py exports CS2680_BASE_URL


def get_base_url() -> str:
    """The course API endpoint: the CS2680_BASE_URL env var (set by madsOpt.py), else the default."""
    return os.environ.get("CS2680_BASE_URL", "").strip() or CS2680_BASE_URL

# Models: one environment variable per tier, exported on the host and forwarded into the
# agent container by run_all.py, holding the exact id the course API expects:
#     CS2680_MODEL_EXPERT     (Expert tier,   course API id "expert")
#     CS2680_MODEL_STANDARD   (Standard tier, course API id "standard")
#     CS2680_MODEL_STARTER    (Starter tier,  course API id "starter")
# If a variable is missing, model_id() falls back to the documented course API id (the alias
# itself) instead of crashing the run. Pick the main tier with MADSOPT_MODEL (default:
# standard). Code that wants a specific tier calls model_id("starter") directly.
MODEL_ENV_VARS = {
    "expert": "CS2680_MODEL_EXPERT",
    "standard": "CS2680_MODEL_STANDARD",
    "starter": "CS2680_MODEL_STARTER",
}
DEFAULT_MODEL_ALIAS = "standard"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# --- harness tunables -------------------------------------------------------------------
PARALLEL = _env_int("MADSOPT_PARALLEL", 5)                 # tasks worked on at once (<= 5 live)
MAX_ITERATIONS = _env_int("MADSOPT_MAX_ITERATIONS", 60)    # model turns for the first grading attempt
RETRY_ITERATIONS = _env_int("MADSOPT_RETRY_ITERATIONS", 18)  # extra turns after a failed grading
FIX_TURNS = _env_int("MADSOPT_FIX_TURNS", 5)               # extra turns when the compile check fails at the end
WRAPUP_TURNS = _env_int("MADSOPT_WRAPUP_TURNS", 5)         # the wrap-up notice comes this many turns before the end
WRAPUP_SECONDS = _env_int("MADSOPT_WRAPUP_SECONDS", 240)   # ... or this many seconds before the time budget ends
CHECKPOINT_FRACTION = 0.45                                 # requirement checklist checkpoint, fraction of attempt 1
EDIT_NUDGE_FRACTION = 0.25                                 # no file edited by this fraction of attempt 1 -> nudge
MAX_ATTEMPTS = _env_int("MADSOPT_MAX_ATTEMPTS", 2)         # gradings per task (1 = no retry)
TASK_BUDGET_S = _env_int("MADSOPT_TASK_BUDGET_S", 2000)    # wall-clock seconds per task, all attempts included
RUN_LIMIT_S = _env_int("MADSOPT_RUN_LIMIT_S", 4 * 3600)    # the dispatcher's whole-run limit (4 h)
RUN_MARGIN_S = _env_int("MADSOPT_RUN_MARGIN_S", 480)       # stop model calls this long before the run limit
RETRY_MIN_S = _env_int("MADSOPT_RETRY_MIN_S", 300)         # a retry needs at least this much time left
REASONING_EFFORT = os.environ.get("MADSOPT_REASONING_EFFORT", "").strip()
RETRY_MODEL_ALIAS = os.environ.get("MADSOPT_RETRY_MODEL", "").strip()   # e.g. "expert": tier for attempt 2+

API_TIMEOUT_S = _env_int("MADSOPT_API_TIMEOUT_S", 600)
API_MAX_RETRIES = _env_int("MADSOPT_API_MAX_RETRIES", 6)
MAX_COMPLETION_TOKENS = _env_int("MADSOPT_MAX_COMPLETION_TOKENS", 0)    # 0 = let the API decide

BUILD_CHECK_TIMEOUT_S = _env_int("MADSOPT_BUILD_CHECK_TIMEOUT_S", 480)  # go build ./... at submit time
TOOL_OUTPUT_CHARS = _env_int("MADSOPT_TOOL_OUTPUT_CHARS", 7000)         # per bash result
READ_OUTPUT_CHARS = _env_int("MADSOPT_READ_OUTPUT_CHARS", 30000)        # per read_file result
CONTEXT_CHAR_BUDGET = _env_int("MADSOPT_CONTEXT_CHARS", 100000)         # compaction threshold (~25k tokens)
KEEP_RECENT_ROUNDS = _env_int("MADSOPT_KEEP_RECENT_ROUNDS", 5)         # rounds never compacted
TRACE_RESULT_CHARS = 20000                                              # tool results in the trace


def get_api_key() -> str:
    """Env var first, then the macro above. Raises if neither is set."""
    key = os.environ.get("CS2680_API_KEY") or CS2680_API_KEY
    if not key:
        raise RuntimeError(
            "No API key: set the CS2680_API_KEY environment variable "
            "(or fill in CS2680_API_KEY in src/config.py for local dev)."
        )
    return key


def model_id(alias: str) -> str:
    """The course API model id for an alias (expert / standard / starter), read from the
    environment variable the host exported. Raises if the alias is unknown; if the
    variable is missing, falls back to the course API's documented id (the alias itself)."""
    var = MODEL_ENV_VARS.get((alias or "").strip().lower())
    if var is None:
        raise RuntimeError(
            f"unknown model alias {alias!r}; allowed: {', '.join(MODEL_ENV_VARS)} "
            "(set MADSOPT_MODEL to one of them)"
        )
    mid = os.environ.get(var, "").strip()
    if not mid:
        mid = alias.strip().lower()           # the course API's documented id for this tier
        if var not in _WARNED_MODEL_VARS:
            _WARNED_MODEL_VARS.add(var)
            print(f"[madsOpt] {var} is not set; using the course API id {mid!r}", file=sys.stderr, flush=True)
    return mid


def get_model_id() -> str:
    """The model for your agent: the alias in MADSOPT_MODEL (default standard)."""
    return model_id(os.environ.get("MADSOPT_MODEL", DEFAULT_MODEL_ALIAS))


def retry_model_id() -> str:
    """The model for grading attempts after the first (MADSOPT_RETRY_MODEL, default: same)."""
    if RETRY_MODEL_ALIAS:
        try:
            return model_id(RETRY_MODEL_ALIAS)
        except RuntimeError:
            pass
    return get_model_id()
