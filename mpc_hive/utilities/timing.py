import ctypes
import errno
import time


class _Timespec(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_nsec", ctypes.c_long),
    ]


_CLOCK_MONOTONIC = 1

try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _CLOCK_NANOSLEEP = _LIBC.clock_nanosleep
    _CLOCK_NANOSLEEP.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_Timespec),
        ctypes.POINTER(_Timespec),
    ]
    _CLOCK_NANOSLEEP.restype = ctypes.c_int
except (AttributeError, OSError):
    _CLOCK_NANOSLEEP = None


def high_resolution_sleep_ns(nsecs: int) -> None:
    if nsecs <= 0:
        return

    nsecs = int(nsecs)
    if _CLOCK_NANOSLEEP is None:
        time.sleep(nsecs / 1_000_000_000.0)
        return

    req = _Timespec(nsecs // 1_000_000_000, nsecs % 1_000_000_000)
    rem = _Timespec()
    while True:
        ret = _CLOCK_NANOSLEEP(
            _CLOCK_MONOTONIC,
            0,
            ctypes.byref(req),
            ctypes.byref(rem),
        )
        if ret == 0:
            return
        if ret != errno.EINTR:
            time.sleep((req.tv_sec * 1_000_000_000 + req.tv_nsec) / 1_000_000_000.0)
            return
        req = _Timespec(rem.tv_sec, rem.tv_nsec)


def high_resolution_sleep_s(seconds: float) -> None:
    high_resolution_sleep_ns(int(seconds * 1_000_000_000))
