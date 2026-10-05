#!/usr/bin/env python3
"""Local end-to-end dry run of the harness, without Docker.

    python3 tests/dry_run.py --fake            # scripted fake model (native tool calls)
    python3 tests/dry_run.py --fake --text     # fake model that rejects tools -> text protocol
    python3 tests/dry_run.py --fake --lazy     # fake model that never submits -> auto-submit / retry / compile-fix
    python3 tests/dry_run.py --fake --explore  # fake model that only explores -> early-edit nudge, then fixes
    python3 tests/dry_run.py --real            # the course API (needs CS2680_API_KEY; uses tier $MADSOPT_MODEL)

It creates a tiny git repository with a bug, starts src/sandbox_server.py on it, fakes the
dispatcher (extract/evaluate/continue/done, grading by running the repo's unittest file) and runs
Agent(dispatcher).run(). The trace lands in a temp dir; the final diff is printed.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

from evaluation_scripts.trace_logger import TraceLogger   # noqa: E402

BUGGY = '''"""Tiny calculator."""


def add(a, b):
    return a - b          # BUG: should add


def mean(xs):
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)
'''
TEST = '''import unittest
import calc


class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(calc.add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(calc.add(-1, 1), 0)

    def test_mean(self):
        self.assertEqual(calc.mean([1, 2, 3]), 2)

    def test_mean_empty_message(self):
        with self.assertRaises(ValueError) as cm:
            calc.mean([])
        self.assertEqual(str(cm.exception), "cannot compute the mean of an empty list")


if __name__ == "__main__":
    unittest.main()
'''
TASK = {
    "problem_statement": "calc.add returns the difference instead of the sum, and mean([]) raises the wrong message.",
    "requirements": "- `add(a, b)` must return a + b.\n- `mean([])` must raise ValueError with the exact message `cannot compute the mean of an empty list`.",
    "interface": "No new interfaces are introduced.",
}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_repo(path: str):
    os.makedirs(path)
    with open(os.path.join(path, "calc.py"), "w") as f:
        f.write(BUGGY)
    with open(os.path.join(path, "test_calc.py"), "w") as f:
        f.write(TEST)
    git = ["git", "-C", path, "-c", "user.email=a@b", "-c", "user.name=dry"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)


class FakeDispatcher:
    def __init__(self, repo: str, url: str, log_dir: str):
        self.repo, self.url, self.log_dir = repo, url, log_dir
        self.tasks = [dict(TASK, task=0, sandbox_url=url, workdir=repo)]
        self.calls = []
        self.patch = ""
        self._loggers = {}

    def logger(self, k):
        if k not in self._loggers:
            os.chdir(self.log_dir)
            self._loggers[k] = TraceLogger(True)
        return self._loggers[k]

    def next_task(self):
        self.calls.append("next_task")
        return self.tasks.pop(0) if self.tasks else None

    def extract_patch(self, k):
        r = subprocess.run(["git", "-C", self.repo, "add", "-N", "."], capture_output=True)
        r = subprocess.run(["git", "-C", self.repo, "diff"], capture_output=True, text=True)
        self.patch = r.stdout
        self.calls.append(f"extract({len(self.patch)}b)")
        return len(self.patch.encode())

    def evaluate(self, k):
        r = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_calc"], cwd=self.repo,
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        total = int((re.search(r"Ran (\d+) tests?", out) or [0, "0"])[1])
        failed = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", out))
        self.calls.append(f"evaluate -> {failed}/{total}")
        print(f"[dry-run] evaluate: {failed}/{total} failing", file=sys.stderr)
        return {"tests_failed": failed, "tests_total": total, "attempt": 0}

    def continue_task(self, k):
        self.calls.append("continue")
        self.patch = ""

    def done(self, k, reason="done", iterations=None):
        self.calls.append(f"done({reason},{iterations})")
        print(f"[dry-run] done: {reason}, {iterations} iterations", file=sys.stderr)
        return {"tests_failed": 0, "tests_total": 0}


# --- a scripted fake model (native tool calls, or text protocol if `tools` is rejected) ----

class ToolsRejected(Exception):
    status_code = 400

    def __str__(self):
        return "Unsupported parameter: 'tools' is not supported by this model"


def _obj(**kw):
    return types.SimpleNamespace(**kw)


SCRIPT = [
    [("bash", {"command": "grep -n 'def ' calc.py && ls"}), ("read_file", {"path": "calc.py"})],
    [("edit_file", {"path": "calc.py", "old_string": "    return a - b          # BUG: should add",
                    "new_string": "    return a + b"})],
    [("bash", {"command": "python3 -m unittest -q test_calc 2>&1 | tail -5", "timeout_s": 60})],
    None,                                             # text-only reply -> the harness nudges
    [("submit", {"summary": "fixed add"})],           # graded: the message test still fails -> retry
    [("edit_file", {"path": "calc.py", "old_string": 'raise ValueError("mean of empty list")',
                    "new_string": 'raise ValueError("cannot compute the mean of an empty list")'})],
    [("submit", {"summary": "fixed add and the mean error message"})],
]


SCRIPT_LAZY = [                                       # never calls submit (MAX_ITERATIONS=5, RETRY=4, FIX=2)
    [("bash", {"command": "grep -n 'def ' calc.py"})],
    [("edit_file", {"path": "calc.py", "old_string": "    return a - b          # BUG: should add",
                    "new_string": "    return a + b"})],
    [("bash", {"command": "ls"})], [("bash", {"command": "ls"})],
    [("bash", {"command": "ls"})],                   # turn 5: auto-submit -> 1/4 failing -> retry (+4 turns)
    [("edit_file", {"path": "calc.py", "old_string": 'raise ValueError("mean of empty list")',
                    "new_string": 'raise ValueError("cannot compute the mean of an empty list"'})],   # syntax error
    [("bash", {"command": "ls"})], [("bash", {"command": "ls"})],
    [("bash", {"command": "ls"})],                   # turn 9: auto-submit -> compile check fails -> +2 fix turns
    [("edit_file", {"path": "calc.py", "old_string": 'an empty list"', "new_string": 'an empty list")'})],
    [("bash", {"command": "ls"})],                   # turn 11: auto-submit -> 0/4 -> solved
]


SCRIPT_EXPLORE = [                                    # no edit for 4 turns (MAX_ITERATIONS=12 -> nudge before turn 4)
    [("bash", {"command": "ls"})],
    [("bash", {"command": "grep -n 'def ' calc.py"})],
    [("read_file", {"path": "calc.py"})],
    [("bash", {"command": "grep -rn mean ."})],
    [("edit_file", {"path": "calc.py", "old_string": "    return a - b          # BUG: should add",
                    "new_string": "    return a + b"})],
    [("edit_file", {"path": "calc.py", "old_string": 'raise ValueError("mean of empty list")',
                    "new_string": 'raise ValueError("cannot compute the mean of an empty list")'})],
    [("submit", {"summary": "fixed add and the mean error message"})],
]


class FakeCompletions:
    def __init__(self, reject_tools: bool, script=None):
        self.n = 0
        self.reject_tools = reject_tools
        self.script = script or SCRIPT

    def create(self, **kwargs):
        if self.reject_tools and "tools" in kwargs:
            raise ToolsRejected()
        step = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        usage = _obj(prompt_tokens=1000 + 10 * self.n, completion_tokens=50, total_tokens=1050 + 10 * self.n)
        if step is None:
            return _obj(choices=[_obj(message=_obj(content="The add bug is fixed; now the message.", tool_calls=None))], usage=usage)
        if "tools" in kwargs:
            calls = [_obj(id=f"c{self.n}_{i}", function=_obj(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(step)]
            return _obj(choices=[_obj(message=_obj(content="", tool_calls=calls))], usage=usage)
        text = "\n".join(f"<tool_call>\n{json.dumps({'name': n, 'arguments': a})}\n</tool_call>" for n, a in step)
        return _obj(choices=[_obj(message=_obj(content=text, tool_calls=None))], usage=usage)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--text", action="store_true", help="with --fake: the model rejects native tools")
    ap.add_argument("--lazy", action="store_true", help="with --fake: the model never calls submit")
    ap.add_argument("--explore", action="store_true", help="with --fake: the model explores without editing until nudged")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    if not (args.fake or args.real):
        ap.error("choose --fake or --real")
    for var, default in (("CS2680_MODEL_EXPERT", "expert"), ("CS2680_MODEL_STANDARD", "standard"), ("CS2680_MODEL_STARTER", "starter")):
        os.environ.setdefault(var, default)
    if args.fake:
        os.environ.setdefault("CS2680_API_KEY", "fake")
    os.environ.setdefault("MADSOPT_TASK_BUDGET_S", "600")
    if args.lazy:
        for var, val in (("MADSOPT_MAX_ITERATIONS", "5"), ("MADSOPT_RETRY_ITERATIONS", "4"),
                         ("MADSOPT_FIX_TURNS", "2"), ("MADSOPT_WRAPUP_TURNS", "2")):
            os.environ[var] = val
    elif args.explore:
        os.environ["MADSOPT_MAX_ITERATIONS"] = "12"
    else:
        os.environ.setdefault("MADSOPT_MAX_ITERATIONS", "20")

    from src import llm
    from src.agentic_loop import Agent
    if args.fake:
        script = SCRIPT_LAZY if args.lazy else SCRIPT_EXPLORE if args.explore else None
        fake = _obj(chat=_obj(completions=FakeCompletions(args.text, script)))
        orig_init = llm.ModelClient.__init__
        llm.ModelClient.__init__ = lambda self, model_id, logger, tools, client=None: orig_init(self, model_id, logger, tools, client=fake)

    tmp = tempfile.mkdtemp(prefix="a3dry_")
    repo = os.path.join(tmp, "repo")
    make_repo(repo)
    port = free_port()
    server = subprocess.Popen([sys.executable, "-B", os.path.join(ROOT, "src", "sandbox_server.py"),
                               "--root", repo, "--port", str(port), "--work", os.path.join(tmp, "work")], stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        disp = FakeDispatcher(repo, url, tmp)
        t0 = time.time()
        Agent(disp).run()
        print(f"\n[dry-run] dispatcher calls: {disp.calls}")
        print(f"[dry-run] {time.time() - t0:.1f}s; final diff:\n{disp.patch}")
        traces = [os.path.join(tmp, "madsOpt_logs", f) for f in os.listdir(os.path.join(tmp, "madsOpt_logs"))]
        events = [json.loads(l)["event"] for p in traces for l in open(p)]
        print(f"[dry-run] trace events: { {e: events.count(e) for e in sorted(set(events))} }")
        ok = any(c.startswith("evaluate -> 0/") for c in disp.calls) and any(c.startswith("done(solved") for c in disp.calls)
        if args.lazy:
            ok = ok and "evaluate -> 1/4" in disp.calls and "continue" in disp.calls and "done(solved,11)" in disp.calls
        if args.explore:
            ok = ok and "done(solved,7)" in disp.calls
        print("[dry-run] RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        server.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
