"""Adaptive concurrency for replay fetching.

Fetching is latency-bound, not bandwidth-bound: a replay is 1.30 MB on the wire
(16x gzip) and takes ~2.7-3.1 s, which is ~0.48 MB/s -- far below any modern
link. So concurrency is the entire lever, and it costs no CPU, which matters
because the engine side of this machine is capped at 3 workers.

But a fixed high concurrency is how you get rate-limited, and 429 storms have
already cost this project a release (all 8 tape fetches failed on 2026-08-12
20:52 and the cycle HELD with nothing built). So the width is not a constant:
it ramps up while requests succeed and halves on a burst of failures, which
finds the server's knee instead of guessing it.

    pool = AdaptivePool(start=8, ceiling=32)
    for eid in ids:
        pool.acquire()                  # blocks until there is room
        ... submit work ...
    pool.report(ok=True)                # per completion

The controller is deliberately simple and observable -- `pool.stats()` prints
what it learned, so a bad ceiling shows up in the log rather than as a mystery
slowdown.
"""
import threading
import time

# A failure rate above this in the trailing window triggers a halving.
FAIL_RATE_TRIP = 0.20
# Consecutive clean completions needed before widening again.
CLEAN_TO_WIDEN = 12
TRAILING = 25


def looks_throttled(exc_or_msg):
    """Is this failure a rate limit rather than a genuine error?

    404 is PERMANENT (the file is gone from that daily dataset) and must not be
    read as throttling, or the controller would narrow itself over data that
    simply does not exist.
    """
    s = str(exc_or_msg)
    if "404" in s or "Not Found" in s:
        return False
    return any(k in s for k in ("429", "Too Many Requests", "rate limit",
                                "RateLimit", "throttl", "Forbidden", "503",
                                "ConnectionAborted", "10053", "timed out",
                                "TimeoutError"))


class AdaptivePool:
    """Semaphore whose width moves with observed success."""

    def __init__(self, start=8, ceiling=32, floor=2, verbose=True):
        self.width = max(floor, min(start, ceiling))
        self.ceiling = ceiling
        self.floor = floor
        self.verbose = verbose
        self._sem = threading.Semaphore(self.width)
        self._lock = threading.Lock()
        self._recent = []              # trailing outcomes, True = ok
        self._debt = 0                 # permits owed back to a narrow()
        self._clean = 0
        self._widened = 0
        self._narrowed = 0
        self._throttles = 0
        self._t0 = time.time()
        self._done = 0

    # -- capacity -----------------------------------------------------------
    def acquire(self):
        self._sem.acquire()

    def release(self):
        # Honour any outstanding narrow() debt instead of handing the permit
        # back. This is what makes narrowing effective while every slot is busy.
        with self._lock:
            if self._debt > 0:
                self._debt -= 1
                return
        self._sem.release()

    def _widen(self):
        if self.width >= self.ceiling:
            return
        step = min(2, self.ceiling - self.width)
        self.width += step
        for _ in range(step):
            self._sem.release()        # hand out the new slots
        self._widened += 1
        if self.verbose:
            print(f"    fetch width -> {self.width} (clean run)", flush=True)

    def _narrow(self):
        if self.width <= self.floor:
            return
        target = max(self.floor, self.width // 2)
        want = self.width - target
        # Reclaim permits without blocking, so a narrow never deadlocks behind
        # in-flight work. BUG FIXED 2026-08-13: whatever could not be reclaimed
        # now was simply lost, so under FULL load -- every permit held, which is
        # exactly when throttling happens -- narrowing reclaimed zero and
        # silently did nothing. The shortfall is now carried as DEBT and
        # swallowed by the next releases, so the narrow always takes effect.
        taken = 0
        for _ in range(want):
            if self._sem.acquire(blocking=False):
                taken += 1
        self._debt += want - taken
        self.width -= want
        self._narrowed += 1
        if self.verbose:
            print(f"    fetch width -> {self.width} (throttled)", flush=True)

    # -- feedback -----------------------------------------------------------
    def report(self, ok, error=None):
        with self._lock:
            self._done += 1
            throttled = (not ok) and looks_throttled(error)
            if throttled:
                self._throttles += 1
            self._recent.append(ok)
            if len(self._recent) > TRAILING:
                del self._recent[:len(self._recent) - TRAILING]
            if ok:
                self._clean += 1
            else:
                self._clean = 0
            fails = sum(1 for r in self._recent if not r)
            rate = fails / len(self._recent)
            if throttled and rate >= FAIL_RATE_TRIP:
                self._narrow()
                self._recent.clear()
            elif self._clean >= CLEAN_TO_WIDEN:
                self._widen()
                self._clean = 0

    def stats(self):
        dt = max(1e-9, time.time() - self._t0)
        return {"width": self.width, "debt": self._debt,
                "completed": self._done,
                "throttles": self._throttles, "widened": self._widened,
                "narrowed": self._narrowed,
                "per_sec": round(self._done / dt, 2)}

    def log_stats(self, label="fetch"):
        s = self.stats()
        print(f"  {label}: {s['completed']} fetches at {s['per_sec']}/s, "
              f"final width {s['width']} "
              f"(widened {s['widened']}x, narrowed {s['narrowed']}x, "
              f"{s['throttles']} throttle signal(s))", flush=True)
