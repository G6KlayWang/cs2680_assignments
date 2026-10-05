"""The tools the model can call, and their execution against a task's Sandbox.

Five tools: bash, read_file, edit_file, write_file, submit. `submit` is handled by the solver
(it triggers patch extraction and grading); the other four run here. Every result is bounded in
size so that a careless command cannot blow up the context (and the bill).
"""

import re

from . import config
from .sandbox import SandboxError, SandboxPathError

READ_DEFAULT_LINES = 400
READ_MAX_LINES = 1000
BASH_DEFAULT_TIMEOUT_S = 120
BASH_MAX_TIMEOUT_S = 900
BUILD_TEST_TIMEOUT_S = 600      # default (and floor) for commands that build or run tests
BUILD_TEST_RE = re.compile(
    r"(^|[\s;&|(])(go\s+(build|test|vet|run|generate|mod\s+(tidy|vendor|download)|get)|pytest|py\.test|tox\b|"
    r"python[0-9.]*\s+-m\s+(pytest|unittest)|jest|vitest|mocha|(yarn|npm|pnpm)\s+(run\s+)?(test|build|lint|tsc|jest)|"
    r"npx\s+(jest|tsc|vitest)|tsc\b|make\b|cargo\s+(build|test)|mvn\b|gradle\w*\b)")

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": (
            "Run a shell command (sh -c) in the repository root inside the task container. Use it to "
            "explore (ls, find, grep -rn), build and run tests. stdout and stderr are merged and the "
            "result is truncated to a few thousand characters, so narrow output with head/tail/grep -n. "
            "Default timeout 120 s, or 600 s for commands that build or run tests (never less than 300 for those); "
            "timeout_s can raise it up to 900."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "the shell command"},
            "timeout_s": {"type": "integer", "description": "seconds before the command is killed (default 120, max 900)"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": (
            "Read a text file with line numbers. `path` is relative to the repository root (or absolute, "
            "inside the repository or under /work). Optional 1-based inclusive range start_line..end_line; "
            "without a range the first 400 lines are returned and the total line count is reported."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": (
            "Edit a file by replacing exactly one occurrence of old_string with new_string. old_string "
            "must match the file text exactly (whitespace and indentation included) and must be unique "
            "in the file: include enough surrounding lines to make it unique. Use an empty new_string "
            "to delete. For new files use write_file."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": ("Create or overwrite a file with the given full content (parent directories are created). "
                        "Paths are inside the repository, or under /work for scratch files that must not end up in the patch."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": (
            "Call when the fix is complete and verified. The harness takes the uncommitted `git diff` of "
            "the repository as your patch and grades it with the hidden tests. Before calling: the code "
            "builds, relevant tests pass, scratch files are removed and `git status` shows only intended "
            "changes."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "what you changed and how you verified it"}},
            "required": ["summary"]}}},
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}


def truncate(text: str, limit: int) -> str:
    """Keep the head and the tail of an over-long text, with a marker in between."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    cut = len(text) - head - tail
    return (text[:head] + f"\n\n...[{cut} characters truncated; narrow the output with head/tail/grep -n "
            f"or a line range to see more]...\n\n" + text[-tail:])


def _numbered(lines, lo: int, hi: int) -> str:
    """Lines lo..hi-1 (0-based) rendered as `NNNN| text`."""
    return "\n".join(f"{i + 1:>5}| {lines[i]}" for i in range(lo, hi))


class ToolExecutor:
    """Runs a tool call against one task's Sandbox. run() returns (result_text, is_error)."""

    def __init__(self, sandbox):
        self.sb = sandbox

    def run(self, name: str, args: dict):
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return f"error: unknown tool {name!r}; available: {', '.join(sorted(TOOL_NAMES))}", True
        if not isinstance(args, dict):
            return f"error: arguments of {name} must be a JSON object", True
        try:
            return fn(**args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}", True
        except SandboxPathError as e:
            return f"error: {e}", True
        except SandboxError as e:
            return f"error: the sandbox is unreachable: {e}", True
        except (ValueError, OverflowError) as e:
            return f"error: {e}", True

    # -- bash ---------------------------------------------------------------------------
    def _t_bash(self, command: str, timeout_s=None):
        try:
            timeout = int(timeout_s) if timeout_s else BASH_DEFAULT_TIMEOUT_S
        except (TypeError, ValueError):
            timeout = BASH_DEFAULT_TIMEOUT_S
        if not isinstance(command, str) or not command.strip():
            return "error: empty command", True
        if BUILD_TEST_RE.search(command):           # builds and test runs are slow; a short timeout wastes the turn
            timeout = BUILD_TEST_TIMEOUT_S if not timeout_s else max(timeout, BUILD_TEST_TIMEOUT_S // 2)
        timeout = max(1, min(timeout, BASH_MAX_TIMEOUT_S))
        r = self.sb.exec(command, timeout_s=timeout)
        out = truncate(r.get("output") or "", config.TOOL_OUTPUT_CHARS)
        code = r.get("exit_code")
        head = f"exit_code: {code}"
        if r.get("timed_out"):
            head += (f"  (TIMED OUT after {timeout}s; the process group was killed. Re-run with timeout_s=900 AND a narrower "
                     f"scope: build or test only the changed package or a single test (go test ./pkg/ -run 'TestName', "
                     f"pytest path::test, jest path). Never lower the timeout.)")
        if not out.strip():
            out = "(no output)"
        return f"{head}\n{out}", bool(r.get("timed_out")) or code != 0

    # -- read_file ----------------------------------------------------------------------
    def _t_read_file(self, path: str, start_line=None, end_line=None):
        text = self.sb.read_text(path)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        n = len(lines)
        if n == 0:
            return f"{path} is empty (0 lines)", False
        start = max(1, int(start_line or 1))
        if start > n:
            return f"error: {path} has only {n} lines (start_line={start})", True
        if end_line:
            end = min(n, int(end_line), start + READ_MAX_LINES - 1)
        else:
            end = min(n, start + READ_DEFAULT_LINES - 1)
        if end < start:
            end = start
        body = _numbered(lines, start - 1, end)
        body = truncate(body, config.READ_OUTPUT_CHARS)
        header = f"{path} (lines {start}-{end} of {n})"
        if end < n:
            header += f" — {n - end} more lines; call again with start_line={end + 1}"
        return f"{header}\n{body}", False

    # -- edit_file ----------------------------------------------------------------------
    def _t_edit_file(self, path: str, old_string: str, new_string: str):
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            return "error: old_string and new_string must be strings", True
        if old_string == "":
            return "error: old_string is empty; to create a file use write_file", True
        text = self.sb.read_text(path)
        count = text.count(old_string)
        if count == 0:
            hint = ""
            stripped = "\n".join(l.rstrip() for l in old_string.split("\n"))
            if stripped != old_string and text.count(stripped) == 1:
                hint = " (a match exists if trailing whitespace is ignored: copy the exact lines from read_file)"
            elif old_string.strip() and text.count(old_string.strip()) >= 1:
                hint = " (the text exists with different leading/trailing whitespace or indentation)"
            return f"error: old_string was not found in {path}{hint}. Re-read the file and copy the text exactly.", True
        if count > 1:
            return f"error: old_string occurs {count} times in {path}; include more surrounding context so it is unique.", True
        pos = text.index(old_string)
        new_text = text[:pos] + new_string + text[pos + len(old_string):]
        self.sb.write_text(path, new_text)
        lines = new_text.split("\n")
        first = new_text.count("\n", 0, pos)
        last = first + new_string.count("\n")
        lo, hi = max(0, first - 3), min(len(lines), last + 4)
        return f"edited {path}; lines {lo + 1}-{hi} now read:\n{_numbered(lines, lo, hi)}", False

    # -- write_file ---------------------------------------------------------------------
    def _t_write_file(self, path: str, content: str):
        if not isinstance(content, str):
            return "error: content must be a string", True
        existed = self.sb.write_text(path, content)
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return f"{'overwrote' if existed else 'created'} {path} ({len(content)} chars, {n_lines} lines)", False
