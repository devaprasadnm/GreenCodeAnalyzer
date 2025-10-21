import psutil
import time
import tracemalloc
import cProfile
import io
import pstats

class DynamicProfiler:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def profile(self):
        process = psutil.Process()
        mem_before = process.memory_info().rss
        tracemalloc.start()

        pr = cProfile.Profile()
        pr.enable()
        start = time.time()

        result = self.func(*self.args, **self.kwargs)

        end = time.time()
        pr.disable()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_after = process.memory_info().rss
        cpu_time = end - start
        mem_used = mem_after - mem_before

        # Profile stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(10)
        stats_str = s.getvalue()

        return {
            "cpu_time": cpu_time,
            "mem_used": mem_used,
            "peak_tracemalloc": peak,
            "profile_stats": stats_str,
        }, result
