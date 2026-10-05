"""Compile checks the harness runs before a patch is graded: a compile error fails every hidden
test, so a submission that does not compile is sent back to the model instead.

- Go repositories: `go build ./...`, compared with a baseline build of the untouched tree that
  starts in the background when the task opens (so a package that never built is not blamed on
  the model). `go vet` of the changed packages (which also compiles their tests) is advisory.
- Python files: a syntax check (ast.parse, which writes no bytecode) of every changed .py file.
- TypeScript/JavaScript: skipped (a full `tsc` run is too slow in the sandbox).
"""

import os
import re
import shlex
import threading

from . import config
from .sandbox import SandboxError

GO_BUILD_CMD = 'out=$(go build ./... 2>&1); rc=$?; printf "%s\\n" "$out" | tail -60; echo "__rc=$rc"'
GO_VET_CMD = 'out=$(go vet {pkgs} 2>&1); rc=$?; printf "%s\\n" "$out" | tail -40; echo "__rc=$rc"'
PY_COMPILE_CMD = ('PY=$(command -v python3 || command -v python); [ -n "$PY" ] || {{ echo "__rc=0"; exit 0; }}; '
                  'out=$("$PY" -c "import ast, sys\n'
                  'for f in sys.argv[1:]:\n'
                  '    ast.parse(open(f, encoding=\\"utf-8\\", errors=\\"replace\\").read(), f)" {files} 2>&1); '
                  'rc=$?; printf "%s\\n" "$out" | tail -30; echo "__rc=$rc"')   # ast.parse writes no __pycache__
CHANGED_CMD = 'git add -N . >/dev/null 2>&1; git diff --name-only | sort -u'
DETECT_CMD = '[ -f go.mod ] && { echo GO; head -1 go.mod; }; git ls-files -- "*.py" | head -1 | sed "s/^/PY /"'


def _rc(out: str):
    m = re.search(r"__rc=(\d+)\s*$", out)
    return int(m.group(1)) if m else None


def _body(out: str) -> str:
    return re.sub(r"__rc=\d+\s*$", "", out).strip()


def _fail_pkgs(out: str) -> set:
    """Import paths of the packages `go build` reported errors for (`# pkg/path` headers)."""
    return {m.split()[0] for m in re.findall(r"^# (\S+.*)$", out, re.M)}


class BuildChecker:
    def __init__(self, sandbox, log):
        self.sb = sandbox
        self.log = log
        self.has_go = False
        self.has_py = False
        self.module = ""
        self.base_rc = None            # None = unknown (not run, timed out, failed)
        self.base_fail = set()
        self._thread = None

    # -- baseline -------------------------------------------------------------------------
    def start(self):
        try:
            out = self.sb.exec(DETECT_CMD, timeout_s=30).get("output") or ""
        except SandboxError:
            return
        lines = out.splitlines()
        if lines and lines[0].strip() == "GO":
            self.has_go = True
            if len(lines) > 1 and lines[1].startswith("module "):
                self.module = lines[1].split()[1].strip()
        self.has_py = any(l.startswith("PY ") for l in lines)
        if self.has_go:
            self._thread = threading.Thread(target=self._baseline, daemon=True)
            self._thread.start()

    def _baseline(self):
        try:
            r = self.sb.exec(GO_BUILD_CMD, timeout_s=config.BUILD_CHECK_TIMEOUT_S)
        except SandboxError:
            return
        if r.get("timed_out"):
            self.log("baseline go build timed out; the compile check will blame only changed packages")
            return
        out = r.get("output") or ""
        self.base_rc = _rc(out)
        self.base_fail = _fail_pkgs(out)
        self.log(f"baseline go build: rc={self.base_rc}, failing packages={len(self.base_fail)}")

    # -- the check ------------------------------------------------------------------------
    def _pkg_dir(self, pkg: str) -> str:
        if self.module and pkg == self.module:
            return "."
        if self.module and pkg.startswith(self.module + "/"):
            return pkg[len(self.module) + 1:]
        return pkg

    def check(self):
        """Returns (ok, blocking, report). blocking=True means: do not grade, send the report back."""
        try:
            changed = [l.strip() for l in (self.sb.exec(CHANGED_CMD, timeout_s=60).get("output") or "").splitlines() if l.strip()]
        except SandboxError as e:
            return True, False, f"(compile check skipped: {e})"
        reports, ok, blocking = [], True, False
        py_files = [f for f in changed if f.endswith(".py")]
        if py_files:
            r = self.sb.exec(PY_COMPILE_CMD.format(files=" ".join(shlex.quote(f) for f in py_files)), timeout_s=180)
            out = r.get("output") or ""
            if _rc(out) not in (0, None):
                ok, blocking = False, True
                reports.append("Python syntax check FAILED:\n" + _body(out))
        go_files = [f for f in changed if f.endswith(".go") or f in ("go.mod", "go.sum")]
        if self.has_go and go_files:
            if self._thread is not None:
                self._thread.join(timeout=config.BUILD_CHECK_TIMEOUT_S)
            r = self.sb.exec(GO_BUILD_CMD, timeout_s=config.BUILD_CHECK_TIMEOUT_S)
            out = r.get("output") or ""
            if r.get("timed_out"):
                reports.append("`go build ./...` timed out in the compile check (not blocking); build the changed packages yourself.")
            elif _rc(out) not in (0, None):
                ok = False
                fail = _fail_pkgs(out)
                changed_dirs = {os.path.dirname(f) or "." for f in changed if f.endswith(".go")}
                if self.base_rc == 0:
                    blocking = True                       # the base tree built; any failure is new
                elif self.base_rc is None:
                    blocking = not fail or any(self._pkg_dir(p) in changed_dirs for p in fail)
                else:
                    blocking = bool(fail - self.base_fail) or not fail
                tag = "FAILED" if blocking else "fails (same packages as the untouched tree; not blocking)"
                reports.append(f"`go build ./...` {tag}:\n" + _body(out))
            else:
                dirs = sorted({"./" + (os.path.dirname(f) or ".") for f in changed if f.endswith(".go")})
                if dirs:
                    r = self.sb.exec(GO_VET_CMD.format(pkgs=" ".join(shlex.quote(d) for d in dirs)), timeout_s=300)
                    out = r.get("output") or ""
                    if not r.get("timed_out") and _rc(out) not in (0, None):
                        reports.append("Advisory: `go vet` of the changed packages (this also compiles their existing "
                                       "tests) reports problems. If you changed a signature, update its callers and "
                                       "existing tests unless the hidden tests replace those files:\n" + _body(out))
        if ok and not reports:
            reports.append("Compile check passed.")
        return ok, blocking, "\n\n".join(reports)
