import time


class Timer:
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.perf_counter() - self.time
        print(f'Time: {self.time:.3f} seconds')
