from __future__ import annotations

import threading
from contextlib import contextmanager


class GpuTaskManager:
    """Serial GPU execution gate for 4GB single-card deployment."""

    def __init__(self, max_concurrent_tasks: int = 1):
        self._sem = threading.Semaphore(max_concurrent_tasks)

    @contextmanager
    def acquire(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


GPU_TASK_MANAGER = GpuTaskManager(max_concurrent_tasks=1)

