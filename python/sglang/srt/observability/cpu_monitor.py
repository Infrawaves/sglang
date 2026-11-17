import os
import threading
import time

import psutil


def start_cpu_monitor_thread(
    component: str, interval: float = 5.0, *, per_worker: bool = False
) -> threading.Thread:
    from prometheus_client import Counter

    labels = {"component": component}
    if per_worker:
        labels["worker_pid"] = str(os.getpid())
    cpu_seconds_total = Counter(
        name="sglang:process_cpu_seconds_total",
        documentation="Total CPU time consumed by this process (user + system)",
        labelnames=labels.keys(),
    ).labels(**labels)

    def monitor():
        process = psutil.Process()
        last_times = process.cpu_times()

        while True:
            time.sleep(interval)
            curr_times = process.cpu_times()
            delta = (curr_times.user - last_times.user) + (
                curr_times.system - last_times.system
            )
            if delta < 0:
                # CRIU restore restarts the process CPU clock near zero. Its
                # current reading is CPU time since restore, not a decrement.
                delta = curr_times.user + curr_times.system
            cpu_seconds_total.inc(delta)
            last_times = curr_times

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return t
