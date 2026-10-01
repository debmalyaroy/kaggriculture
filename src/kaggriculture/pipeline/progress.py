"""Uniform, flushed progress output.

Long downloads were silent: Python line-buffers stdout when it is a pipe, and
on Windows a `python ... > log.txt` run could sit for minutes with nothing on
screen. Everything here flushes on every line, so a run always shows what it is
doing right now.

Every line carries elapsed wall-clock time since the run started:

    [00:03] indexing daily datasets
    [00:11]   2026-08-03: 790 episodes, top score 1204.6
    [01:44]     [ 12/40] 68123456.json  27.1 MB  2.9s  eta 01:22

Set KAGG_VERBOSE=1 (or pass --verbose to download_data.py) to additionally echo
every Kaggle CLI invocation and how long it took.
"""
import os
import sys
import time

_T0 = time.time()

VERBOSE = os.environ.get("KAGG_VERBOSE", "").strip() not in ("", "0", "false", "no")


def set_verbose(on=True):
    """Turn command echoing on for this process and any child it spawns."""
    global VERBOSE
    VERBOSE = bool(on)
    os.environ["KAGG_VERBOSE"] = "1" if on else "0"


def reset():
    """Restart the elapsed-time clock (call at the top of a run)."""
    global _T0
    _T0 = time.time()


def elapsed():
    return time.time() - _T0


def hms(sec):
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def log(msg, indent=0):
    """One progress line, stamped and flushed."""
    print(f"[{hms(elapsed())}] {'  ' * indent}{msg}", flush=True)


def vlog(msg, indent=0):
    """Progress line that only appears in verbose mode."""
    if VERBOSE:
        log(msg, indent)


def warn(msg, indent=0):
    log(f"! {msg}", indent)


def banner(title):
    line = "=" * max(len(title), 60)
    print(line, flush=True)
    print(title, flush=True)
    print(line, flush=True)


class Ticker:
    """Counter with rate and ETA for a loop of known (or guessed) length.

        tick = Ticker(total=40, label="episodes")
        for ...:
            tick.step(f"{fname}  {mb:.1f} MB")
        tick.done()
    """

    def __init__(self, total=None, label="items", indent=2):
        self.total = total
        self.label = label
        self.indent = indent
        self.n = 0
        self.t0 = time.time()
        self.t_last = self.t0

    def step(self, msg="", extra=""):
        self.n += 1
        now = time.time()
        dt = now - self.t_last
        self.t_last = now
        head = f"[{self.n:>3}/{self.total}]" if self.total else f"[{self.n:>3}]"
        eta = ""
        if self.total and self.n < self.total:
            per = (now - self.t0) / self.n
            eta = f"  eta {hms(per * (self.total - self.n))}"
        log(f"{head} {msg}  {dt:.1f}s{eta}{(' ' + extra) if extra else ''}",
            self.indent)

    def skip(self, msg):
        log(f"      - {msg}", self.indent)

    def done(self, msg=""):
        secs = time.time() - self.t0
        rate = f"{self.n / secs:.2f}/s" if secs > 0 else "-"
        log(f"{self.n} {self.label} in {hms(secs)} ({rate}) {msg}".rstrip(),
            self.indent - 1 if self.indent else 0)


def heartbeat(msg, key=None, every=15.0, _state={}):
    """Print `msg` at most once every `every` seconds.

    For loops that would otherwise be silent for long stretches. Rate-limiting
    is keyed on `key`, not on the message -- a message with a counter in it
    changes every call and would defeat the limiter.
    """
    k = key if key is not None else msg
    now = time.time()
    if now - _state.get(k, 0) >= every:
        _state[k] = now
        log(msg, 2)
