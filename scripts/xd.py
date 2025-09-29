import cupy as cp
print("CuPy:", cp.__version__)
print("Runtime:", cp.cuda.runtime.runtimeGetVersion())
x = cp.arange(10, dtype=cp.float32)
print("GPU sum:", x.sum().item())