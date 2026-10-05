"""The agent: `Agent(dispatcher).run()` is the entry point madsOpt.py calls.

Design
- Up to PARALLEL (<= 5, the facility's live-task cap) worker threads. Each worker repeatedly
  opens the next task, solves it with a TaskSolver (src/solver.py) and finalizes it with done().
  next_task() is serialized by a lock, so the live-task cap is never exceeded.
- One shared run deadline, RUN_LIMIT_S after the first task was opened, bounds every solver so
  that each task hands in what it has before the facility stops the run.
- Model calls go through src/llm.py (retries, trace logging), repository access through
  src/tools.py over src/sandbox.py; src/prompts.py holds the prompts; src/config.py the knobs.
"""

import sys
import threading
import time
import traceback

from . import config
from .solver import DispatcherError, DispatcherShutdown, TaskSolver


def _debug(msg: str):
    print(f"[madsOpt] {msg}", file=sys.stderr, flush=True)


class Agent:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.model_id = config.get_model_id()        # fails early if the model env is missing
        self.n_workers = max(1, min(config.PARALLEL, 5))
        self._open_lock = threading.Lock()
        self._stop = threading.Event()
        self._run_start = None

    def run_deadline(self) -> float:
        return (self._run_start or time.time()) + config.RUN_LIMIT_S

    def run(self):
        _debug(f"agent starting: model={self.model_id} parallel={self.n_workers} "
               f"turns/task={config.MAX_ITERATIONS} budget/task={config.TASK_BUDGET_S}s attempts={config.MAX_ATTEMPTS}")
        threads = [threading.Thread(target=self._worker, name=f"worker-{i}", daemon=True)
                   for i in range(self.n_workers)]
        for t in threads:
            t.start()
            time.sleep(1)                      # stagger the first sandbox starts
        for t in threads:
            t.join()
        _debug("agent finished")

    def _open_task(self):
        with self._open_lock:
            if self._stop.is_set():
                return None
            if self._run_start is not None and self.run_deadline() - time.time() < config.RUN_MARGIN_S + 180:
                _debug("the run limit is near; not opening more tasks")
                return None
            task = self.dispatcher.next_task()
            if task is not None and self._run_start is None:
                self._run_start = time.time()
            return task

    def _worker(self):
        name = threading.current_thread().name
        refusals = 0
        while not self._stop.is_set():
            try:
                task = self._open_task()
            except DispatcherShutdown:
                self._stop.set()
                return
            except DispatcherError as e:
                if "too many live" in str(e):   # the facility's cap is below our pool size: this
                    _debug(f"{name}: live-task cap reached; worker exits (active workers continue)")
                    return                      # worker is not needed, the active ones pick up the rest
                refusals += 1
                if refusals > 3:
                    _debug(f"{name}: next_task keeps failing ({e}); worker exits")
                    return
                _debug(f"{name}: next_task refused ({e}); retrying in 20s")
                time.sleep(20)
                continue
            if task is None:
                return
            try:
                TaskSolver(self.dispatcher, task, self.run_deadline).run()
            except DispatcherShutdown:
                self._stop.set()
                return
            except Exception as e:             # noqa: BLE001 — one task must not kill the worker
                _debug(f"{name}: task {task.get('task')} failed outside the solver: "
                       f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
