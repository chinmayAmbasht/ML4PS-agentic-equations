"""Order-preserving parallel map over INDEPENDENT units (v1 Plans/12). Scheduling only -- no numerical effect.

PK_NPAR unset or 1 -> the plain serial list comprehension (the registered path, bitwise unchanged).
PK_NPAR = N > 1    -> a fork Pool of up to N workers. The unit function and its items are placed in a module global before
                      the fork, so workers inherit compiled forms (sympy lambdas are not picklable); only the index goes out
                      and only the result comes back. Results are returned in INPUT ORDER, so every downstream reduction
                      (sums, tie-broken max, sigma_fold) runs over exactly the same ordered values as the serial path.
Inside a worker, pmap is serial (no nested pools).
"""
import multiprocessing as mp
import os

_CTX = None
_IN_WORKER = False


def npar() -> int:
    if _IN_WORKER or mp.current_process().daemon:     # daemonic (outer Pool) workers cannot fork children
        return 1
    try:
        return max(1, int(os.environ.get("PK_NPAR", "1")))
    except ValueError:
        return 1


def _init():
    global _IN_WORKER
    _IN_WORKER = True


def _call(i):
    fn, items = _CTX
    return fn(items[i])


def pmap(fn, items):
    items = list(items)
    n = min(npar(), len(items))
    if n <= 1:
        return [fn(x) for x in items]
    global _CTX
    prev, _CTX = _CTX, (fn, items)
    try:
        with mp.get_context("fork").Pool(n, initializer=_init) as pool:
            return pool.map(_call, range(len(items)), chunksize=1)
    finally:
        _CTX = prev
