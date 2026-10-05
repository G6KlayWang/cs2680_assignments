"""Per-task solver: the agent loop for one task, from the task dict to dispatcher.done().

Lifecycle: repo overview -> tool-using model loop (with a requirement checkpoint and a wrap-up
notice) -> `submit`, or an automatic submission when the attempt's turns or time run out ->
cleanup + compile check -> extract_patch -> evaluate -> solved, or a retry with feedback and a
fresh turn allowance (continue_task), or final. Budgets (turns per attempt, wall clock, the
whole-run deadline) are enforced here; whatever is in the repository is always handed in.
"""

import sys
import time
import traceback

from . import config, prompts
from .checks import BuildChecker
from .llm import ContextLengthError, ModelClient
from .sandbox import Sandbox, SandboxError
from .tools import TOOL_SPECS, ToolExecutor, truncate

try:
    from dispatcher import DispatcherError, DispatcherShutdown
except ImportError:                       # local dry runs outside the harness
    class DispatcherError(Exception):
        pass

    class DispatcherShutdown(Exception):
        pass

CLEANUP_CMD = r"""
find . -name .git -prune -o -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name .hypothesis \) -prune -exec rm -rf {} + 2>/dev/null
find . -name .git -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.orig' -o -name '*.rej' \) -exec rm -f {} + 2>/dev/null
git add -N . >/dev/null 2>&1
git diff --numstat --diff-filter=A | awk '$1=="-" && $2=="-" {print $3}' | while IFS= read -r f; do echo "removed new binary file: $f"; rm -f -- "$f"; done
git diff --numstat --diff-filter=MD | awk '$1=="-" && $2=="-" {print $3}' | while IFS= read -r f; do echo "reverted binary file (binaries cannot be patched): $f"; git checkout -- "$f" 2>/dev/null; done
git add -N . >/dev/null 2>&1
"""
PRESUBMIT_CMD = CLEANUP_CMD + r"""
echo "## git status --short"; git status --short | head -80
echo "## git diff --stat"; git diff --stat | tail -30
echo "## binary files in diff"; git diff --numstat | awk '$1=="-" && $2=="-" {print $3}'
echo "## end"
"""
COMPACT_NOTE = ("\n\n[Earlier exploration steps were removed from this conversation to save context. "
                "Your edits to the repository are intact; use `git diff` to review them.]")
MAX_TEXT_ONLY_TURNS = 3
MAX_CONTEXT_RETRIES = 2
MAX_AUTO_SUBMITS = 4


def _section(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return text[i:j] if j >= 0 else text[i:]


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


class TaskSolver:
    def __init__(self, dispatcher, task: dict, run_deadline):
        self.d = dispatcher
        self.task = task
        self.k = int(task["task"])
        self.logger = dispatcher.logger(self.k)
        self.run_deadline = run_deadline        # callable -> absolute time of the run limit
        self.t0 = time.time()
        self.iteration = 0
        self.attempt = 0
        self.turn_limit = config.MAX_ITERATIONS  # turns allowed in the current attempt (cumulative)
        self.checkpoint_turn = max(3, int(config.MAX_ITERATIONS * config.CHECKPOINT_FRACTION))
        self.checkpoint_sent = False
        self.edit_nudge_turn = max(2, int(config.MAX_ITERATIONS * config.EDIT_NUDGE_FRACTION))
        self.edit_nudge_sent = False
        self.edits = 0                            # successful edit_file / write_file calls
        self.wrapup_sent = False
        self.fix_granted = False
        self.auto_submits = 0
        self.finished = False
        self.messages = []
        self.last_eval = None
        self.sb = self.tools = self.model = self.checker = None

    def log(self, msg: str):
        print(f"[madsOpt] task {self.k}: {msg}", file=sys.stderr, flush=True)

    # -- budgets ------------------------------------------------------------------------
    def deadline(self) -> float:
        return min(self.t0 + config.TASK_BUDGET_S, self.run_deadline() - config.RUN_MARGIN_S)

    def seconds_left(self) -> float:
        return self.deadline() - time.time()

    def turns_left(self) -> int:
        return self.turn_limit - self.iteration

    # -- lifecycle ----------------------------------------------------------------------
    def run(self):
        reason = "error"
        try:
            if not self.task.get("sandbox_url"):
                self.log("no sandbox was started for this task")
                reason = "no-sandbox"
                return
            self.sb = Sandbox(self.task["sandbox_url"], self.task["workdir"])
            self.tools = ToolExecutor(self.sb)
            self.model = ModelClient(config.get_model_id(), self.logger, TOOL_SPECS)
            self.checker = BuildChecker(self.sb, self.log)
            overview = self.repo_overview()
            self.checker.start()
            self.messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                _user(prompts.build_task_prompt(self.task, overview, config.MAX_ITERATIONS,
                                                int(min(config.TASK_BUDGET_S, self.seconds_left()) // 60))),
            ]
            reason = self.loop()
        except DispatcherShutdown:
            self.finished = True              # the facility finalizes every open task itself
            raise
        except SandboxError as e:
            self.log(f"sandbox error: {e}")
            reason = "sandbox-error"
        except Exception as e:                # noqa: BLE001 — a crash must still hand the task in
            self.log(f"crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            reason = f"error:{type(e).__name__}"
        finally:
            self.finish(reason)

    def finish(self, reason: str):
        if self.finished:
            return
        self.finished = True
        try:
            if reason not in ("solved", "unresolved", "no-sandbox") and self.sb is not None:
                try:                          # hand in whatever is in the repository; done() grades it
                    self.sb.exec(CLEANUP_CMD, timeout_s=120)
                    size = self.d.extract_patch(self.k)
                    self.log(f"final patch extracted: {size} bytes")
                except (DispatcherError, SandboxError) as e:
                    self.log(f"final extract failed: {e}")
            if self.model is not None:
                self.log(f"finishing ({reason}): {self.iteration} model calls, "
                         f"{self.model.prompt_tokens} prompt / {self.model.completion_tokens} completion tokens, "
                         f"{int(time.time() - self.t0)}s, {self.attempt} grading(s)")
            self.d.done(self.k, reason, self.iteration)
        except DispatcherShutdown:
            raise
        except DispatcherError as e:
            self.log(f"done() refused: {e}")

    def repo_overview(self) -> str:
        try:
            r = self.sb.exec(prompts.OVERVIEW_CMD, timeout_s=60)
            return truncate(r.get("output") or "", 4000)
        except SandboxError:
            raise
        except Exception as e:                # noqa: BLE001
            return f"(overview failed: {e})"

    # -- the loop -----------------------------------------------------------------------
    def loop(self) -> str:
        text_only = 0
        context_retries = 0
        while True:
            if self.turns_left() <= 0 or self.seconds_left() <= 0:
                verdict = self.auto_submit()
                if verdict in ("continue", "retry"):
                    continue
                return verdict
            self.inject_notes()
            try:
                outcome = self.step()
            except ContextLengthError as e:
                context_retries += 1
                if context_retries > MAX_CONTEXT_RETRIES:
                    self.log(f"context too long even after compaction: {e}")
                    return "context-overflow"
                self.log("context too long; compacting hard")
                self.compact(hard=True)
                continue
            context_retries = 0
            if outcome == "text":
                text_only += 1
                if text_only < MAX_TEXT_ONLY_TURNS:
                    self.messages.append(_user(prompts.NUDGE))
                    continue
                self.log("model stopped calling tools; treating as submit")
                report, err, _ = self.presubmit()
                text_only = 0
                if err:
                    self.messages.append(_user(report + "\n\n" + prompts.NUDGE))
                    continue
                self.messages.append(_user("Harness: your replies contained no tool calls, so the current "
                                           "repository state has been submitted for grading.\n\n" + report))
                outcome = "submit"
            else:
                text_only = 0
            if outcome == "submit":
                verdict = self.grade()
                if verdict != "retry":
                    return verdict

    def inject_notes(self):
        """Harness notices before the next model call: the edit nudge, the checkpoint and the wrap-up."""
        if not self.edit_nudge_sent and self.attempt == 0 and self.edits == 0 and self.iteration >= self.edit_nudge_turn:
            self.edit_nudge_sent = True
            self.log(f"no file edited by turn {self.iteration}; nudging the model to implement")
            self.messages.append(_user(prompts.edit_nudge(self.iteration, self.turn_limit)))
        if not self.checkpoint_sent and self.attempt == 0 and self.iteration >= self.checkpoint_turn:
            self.checkpoint_sent = True
            self.messages.append(_user(prompts.checkpoint(self.iteration, self.turn_limit)))
        left = self.turns_left()
        if not self.wrapup_sent and (left <= config.WRAPUP_TURNS or self.seconds_left() <= config.WRAPUP_SECONDS):
            self.wrapup_sent = True
            self.messages.append(_user(prompts.wrapup(left, self.seconds_left())))

    def step(self) -> str:
        """One model call plus its tool calls. Returns 'submit', 'tools' or 'text'."""
        self.iteration += 1
        turn = self.model.chat(self.messages, self.iteration)
        self.messages.append(turn.message)
        if not turn.tool_calls:
            return "text"
        submit = False
        for i, call in enumerate(turn.tool_calls):
            if call.error:
                result, err = f"error: {call.error}", True
            elif call.name == "submit":
                self.logger.tool_call(self.iteration, "submit", call.arguments)
                result, err, _ = self.presubmit()
                submit = submit or not err
            else:
                self.logger.tool_call(self.iteration, call.name, call.arguments)
                result, err = self.tools.run(call.name, call.arguments)
                if not err and call.name in ("edit_file", "write_file"):
                    self.edits += 1
            self.logger.tool_result(self.iteration, call.name, truncate(result, config.TRACE_RESULT_CHARS), err)
            if i == len(turn.tool_calls) - 1:
                result += prompts.budget_note(self.turns_left(), config.WRAPUP_TURNS)
            self.messages.append(self.model.tool_result_message(call, result))
        self.compact()
        return "submit" if submit else "tools"

    # -- submission and grading -----------------------------------------------------------
    def presubmit(self):
        """Clean the tree, check it, run the compile check. Returns (report, is_error, kind)."""
        r = self.sb.exec(PRESUBMIT_CMD, timeout_s=120)
        out = r.get("output") or ""
        cleaned = out[:out.find("## git status --short")].strip() if "## git status --short" in out else ""
        if cleaned:
            self.log("pre-submit cleanup: " + cleaned.replace("\n", "; "))
            cleaned = "Harness cleanup before grading:\n" + cleaned + "\n\n"
        status = _section(out, "## git status --short", "## git diff --stat").strip()
        if not status:
            return "error: nothing to submit: `git status` shows no changes in the repository.", True, "empty"
        binaries = _section(out, "## binary files in diff", "## end").strip()
        if binaries:
            return (f"error: the diff contains binary files, which cannot be submitted:\n{binaries}\n"
                    f"Delete them (rm) or move them out of the repository, then call submit again."), True, "binary"
        tree = truncate(out[out.find("## git status --short"):out.find("## binary files in diff")].strip(), 5000)
        if self.run_deadline() - time.time() < 600:
            check = "(compile check skipped: the run deadline is near)"
        else:
            t0 = time.time()
            ok, blocking, check = self.checker.check()
            self.log(f"compile check: {'ok' if ok else 'FAILED'}{'' if ok or blocking else ' (not blocking)'} in {int(time.time() - t0)}s")
            if blocking:
                return ("error: the compile check failed; grading now would fail every test. Fix these errors, "
                        f"then call submit again.\n\n{check}"), True, "build"
        return cleaned + "Pre-submit check passed; this diff is being graded by the hidden tests:\n" + tree + "\n\n" + check, False, None

    def auto_submit(self) -> str:
        """The attempt's turns or time ran out: grade the repository as it is."""
        self.auto_submits += 1
        self.log(f"budget of attempt {self.attempt + 1} exhausted at turn {self.iteration} ({int(time.time() - self.t0)}s)")
        if self.auto_submits > MAX_AUTO_SUBMITS:
            return "budget"
        report, err, kind = self.presubmit()
        if err and kind == "empty":
            self.log("nothing to submit")
            return "budget-no-patch"
        if err and kind == "build" and not self.fix_granted and self.seconds_left() > 60:
            self.fix_granted = True
            self.wrapup_sent = True
            self.turn_limit = self.iteration + config.FIX_TURNS
            self.log(f"compile check failed; granting {config.FIX_TURNS} turns to fix it")
            self.messages.append(_user(prompts.build_fail_note(report, config.FIX_TURNS)))
            return "continue"
        self.messages.append(_user(prompts.auto_submit_note(report)))
        return self.grade()

    def grade(self) -> str:
        """extract + evaluate. Returns 'solved', 'unresolved' or 'retry' (feedback appended)."""
        try:
            self.sb.exec(CLEANUP_CMD, timeout_s=120)   # again: checks and test runs may have left caches
            size = self.d.extract_patch(self.k)
        except (DispatcherError, SandboxError) as e:
            self.log(f"extract failed: {e}")
            size = 0
        if not size:
            self.messages.append(_user("The extracted patch is empty (`git diff` produced nothing). Make your changes, then call submit again."))
            return "retry"
        self.attempt += 1
        self.log(f"attempt {self.attempt}: grading a {size}-byte patch after {self.iteration} turns")
        try:
            ev = self.d.evaluate(self.k)
        except DispatcherError as e:
            self.log(f"evaluate refused: {e}")
            self.attempt -= 1
            self.messages.append(_user(f"Grading could not start ({e}). Fix the problem if it is yours, then call submit again."))
            return "retry"
        self.last_eval = ev
        failed, total = int(ev.get("tests_failed", -1)), int(ev.get("tests_total", 0))
        self.log(f"attempt {self.attempt}: {failed}/{total} hidden tests failing")
        if failed == 0:
            return "solved"
        attempts_left = config.MAX_ATTEMPTS - self.attempt
        if attempts_left <= 0 or self.seconds_left() < config.RETRY_MIN_S:
            self.log("no retry: " + ("attempts exhausted" if attempts_left <= 0 else "time budget too low"))
            return "unresolved"
        self.d.continue_task(self.k)
        if config.RETRY_MODEL_ALIAS:
            self.model.set_model(config.retry_model_id())
        self.turn_limit = self.iteration + config.RETRY_ITERATIONS
        self.wrapup_sent = False
        self.fix_granted = False
        self.messages.append(_user(prompts.eval_feedback(failed, total, self.attempt, attempts_left, config.RETRY_ITERATIONS)))
        return "retry"

    # -- context management ---------------------------------------------------------------
    def _size(self) -> int:
        n = 0
        for m in self.messages:
            n += len(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                n += len(tc["function"].get("arguments") or "")
        return n

    def compact(self, hard: bool = False):
        """Shrink old tool results (and, if still too big, drop old rounds) to bound the context."""
        budget = config.CONTEXT_CHAR_BUDGET // (2 if hard else 1)
        if self._size() <= budget:
            return
        before = self._size()
        rounds = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        keep = max(1, config.KEEP_RECENT_ROUNDS // (2 if hard else 1))
        cutoff = rounds[-keep] if len(rounds) > keep else 2
        for m in self.messages[2:cutoff]:
            c = m.get("content") or ""
            is_result = m["role"] == "tool" or (m["role"] == "user" and c.startswith("<tool_result"))
            if is_result and len(c) > 600:
                m["content"] = c[:450] + "\n...[earlier output truncated to save context; re-run the tool if needed]" \
                    + ("\n</tool_result>" if m["role"] == "user" else "")
        if self._size() > budget and len(rounds) > keep + 1:
            drop_to = rounds[-(keep + 1)]
            if drop_to > 2:
                del self.messages[2:drop_to]
                if COMPACT_NOTE not in self.messages[1]["content"]:
                    self.messages[1]["content"] += COMPACT_NOTE
        self.log(f"compacted context {before} -> {self._size()} chars")
