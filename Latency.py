import time
from collections import defaultdict

class LatencyTracker:
    def __init__(self):
        self.timings = defaultdict(list)  # stage_name -> list of durations (seconds)

    def record(self, stage: str, duration: float):
        self.timings[stage].append(duration)

    def time_block(self, stage: str):
        """Context manager: `with tracker.time_block('embedding'): ...`"""
        return _TimedBlock(self, stage)

    def report(self, title: str = "Latency Report"):
        print(f"\n--- {title} ---")
        total = 0.0
        for stage, durations in self.timings.items():
            stage_total = sum(durations)
            total += stage_total
            avg = stage_total / len(durations)
            print(f"{stage:25s} calls={len(durations):3d}  "
                  f"total={stage_total*1000:8.2f} ms  avg={avg*1000:7.2f} ms")
        print(f"{'-'*70}")
        print(f"{'TOTAL':25s} {total*1000:8.2f} ms")

    def reset(self):
        self.timings.clear()


class _TimedBlock:
    def __init__(self, tracker: LatencyTracker, stage: str):
        self.tracker = tracker
        self.stage = stage
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        self.tracker.record(self.stage, duration)