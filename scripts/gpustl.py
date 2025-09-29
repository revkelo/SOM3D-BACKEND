#!/usr/bin/env python
# -*- coding: utf-8 -*-

# bench_cube_gpu_cpu_warm.py (100k por defecto)
# - Calienta la GPU (CuPy)
# - Repite y reporta medianas
# - Replicación vectorizada (sin bucles Python) para escalar a 100k
# - Genera cube_gpu.stl y cube_cpu.stl

import time, struct, sys, argparse, statistics as stats
import numpy as np

# Intentar GPU (CuPy)
_HAS_CUPY = False
try:
    import os
    os.environ.setdefault("CUPY_DONT_WARN_ON_CUDA_PATH", "1")
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

def faces_indices(xp):
    return xp.array([
        [0,1,2],[0,2,3],     # -Z
        [4,6,5],[4,7,6],     # +Z
        [0,3,7],[0,7,4],     # -X
        [1,5,6],[1,6,2],     # +X
        [0,4,5],[0,5,1],     # -Y
        [3,2,6],[3,6,7],     # +Y
    ], dtype=xp.int32)

# --- Escritura STL binaria (rápida, vectorizada) ---
_TRI_DTYPE = np.dtype([
    ('normal',  '<f4', (3,)),
    ('v1',      '<f4', (3,)),
    ('v2',      '<f4', (3,)),
    ('v3',      '<f4', (3,)),
    ('attr',    '<u2')
])

def write_binary_stl(path, normals_np, tris_np):
    N = tris_np.shape[0]
    rec = np.empty(N, dtype=_TRI_DTYPE)
    rec['normal'][:] = normals_np
    rec['v1'][:]     = tris_np[:,0,:]
    rec['v2'][:]     = tris_np[:,1,:]
    rec['v3'][:]     = tris_np[:,2,:]
    rec['attr'][:]   = 0
    with open(path, "wb") as f:
        header = b"bench cube warm" + b"\x00"*(80- len(b"bench cube warm"))
        f.write(header[:80])
        f.write(struct.pack("<I", N))
        f.write(rec.tobytes(order='C'))

# -------- Generación de cubo base --------
def make_cube_cpu(size=50.0, center=(0,0,0)):
    s = float(size)*0.5
    cx, cy, cz = map(float, center)
    V = np.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ], dtype=np.float32)
    F = faces_indices(np)
    tris = np.stack([V[F[:,0]], V[F[:,1]], V[F[:,2]]], axis=1)  # (12,3,3)
    e1 = tris[:,1,:] - tris[:,0,:]
    e2 = tris[:,2,:] - tris[:,0,:]
    n  = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True); norm[norm==0]=1.0
    n = n / norm
    return n.astype(np.float32, copy=False), tris.astype(np.float32, copy=False)

def make_cube_gpu(size=50.0, center=(0,0,0)):
    if not _HAS_CUPY:
        raise RuntimeError("No hay CuPy/CUDA para GPU.")
    s = float(size)*0.5
    cx, cy, cz = map(float, center)
    V = cp.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ], dtype=cp.float32)
    F = faces_indices(cp)
    tris = cp.stack([V[F[:,0]], V[F[:,1]], V[F[:,2]]], axis=1)
    e1 = tris[:,1,:] - tris[:,0,:]
    e2 = tris[:,2,:] - tris[:,0,:]
    n  = cp.cross(e1, e2)
    norm = cp.linalg.norm(n, axis=1, keepdims=True)
    n = n / cp.where(norm == 0, 1.0, norm)
    return n, tris

# -------- Replicación vectorizada (sin bucles) --------
def replicate_mesh_cpu(normals, tris, reps, shift=60.0):
    if reps <= 1: return normals, tris
    base = tris[None, ...]                                        # (1,12,3,3)
    big  = np.broadcast_to(base, (reps,) + base.shape[1:]).copy() # (reps,12,3,3)
    big[:,:,:,0] += (np.arange(reps, dtype=np.float32)[:,None,None] * shift)
    big = big.reshape(-1, 3, 3)                                   # (reps*12,3,3)
    norms = np.repeat(normals, reps, axis=0)                      # (reps*12,3)
    return norms, big

def replicate_mesh_gpu(normals, tris, reps, shift=60.0):
    if reps <= 1: return normals, tris
    base = tris[None, ...]                                        # (1,12,3,3)
    big  = cp.broadcast_to(base, (reps,) + base.shape[1:]).copy() # (reps,12,3,3)
    big[:,:,:,0] += (cp.arange(reps, dtype=cp.float32)[:,None,None] * shift)
    big = big.reshape(-1, 3, 3)                                   # (reps*12,3,3)
    norms = cp.repeat(normals, reps, axis=0)                      # (reps*12,3)
    return norms, big

def median_ms(vals): 
    return f"{stats.median(vals)*1000:.3f} ms"

def bench_gpu(size, center, reps, iters):
    if not _HAS_CUPY:
        return None
    # Warm-up
    _ = cp.zeros((1,), dtype=cp.float32)
    n, t = make_cube_gpu(size, center)
    n, t = replicate_mesh_gpu(n, t, reps)
    cp.cuda.runtime.deviceSynchronize()

    comp, d2h, io, total = [], [], [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        n, t = make_cube_gpu(size, center)
        n, t = replicate_mesh_gpu(n, t, reps)
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter()

        n_np = cp.asnumpy(n).astype(np.float32, copy=False)
        t_np = cp.asnumpy(t).astype(np.float32, copy=False)
        t2 = time.perf_counter()

        write_binary_stl("cube_gpu.stl", n_np, t_np)
        t3 = time.perf_counter()

        comp.append(t1 - t0)
        d2h.append(t2 - t1)
        io.append(t3 - t2)
        total.append(t3 - t0)

    try: cp.get_default_memory_pool().free_all_blocks()
    except Exception: pass

    return {
        "compute": median_ms(comp),
        "d2h": median_ms(d2h),
        "io": median_ms(io),
        "total": median_ms(total),
        "faces": int(t_np.shape[0]),
    }

def bench_cpu(size, center, reps, iters):
    comp, io, total = [], [], []
    faces_out = 0
    for _ in range(iters):
        c0 = time.perf_counter()
        n, t = make_cube_cpu(size, center)
        n, t = replicate_mesh_cpu(n, t, reps)
        c1 = time.perf_counter()
        write_binary_stl("cube_cpu.stl", n, t)
        c2 = time.perf_counter()
        comp.append(c1 - c0)
        io.append(c2 - c1)
        total.append(c2 - c0)
        faces_out = int(t.shape[0])
    return {
        "compute": median_ms(comp),
        "io": median_ms(io),
        "total": median_ms(total),
        "faces": faces_out,
    }

def main():
    ap = argparse.ArgumentParser(description="Benchmark GPU vs CPU (caliente) para cubo STL")
    ap.add_argument("--size", type=float, default=50.0, help="Arista del cubo")
    ap.add_argument("--reps", type=int, default=100_000, help="Repeticiones del cubo (default 100k)")
    ap.add_argument("--iters", type=int, default=5, help="Iteraciones para mediana (default 5)")
    ap.add_argument("--no-save", action="store_true", help="No escribir STL (mide solo compute+D→H)")
    args = ap.parse_args()

    size, center, reps, iters = args.size, (0.0,0.0,0.0), max(1,args.reps), max(3,args.iters)

    print(f"Config → size={size}, reps={reps}, iters={iters} | CuPy={_HAS_CUPY}")
    if _HAS_CUPY:
        # Ejecutar GPU
        g_comp, g_d2h, g_io, g_total, g_faces = [], [], [], [], None
        # Una pasada para obtener datos y opcionalmente escribir
        t0 = time.perf_counter()
        g = bench_gpu(size, center, reps, iters)
        t1 = time.perf_counter()
        if g is not None:
            print("\nGPU (caliente):")
            print(f"  Compute: {g['compute']}")
            print(f"  D→H:     {g['d2h']}")
            print(f"  IO:      {g['io']}")
            print(f"  TOTAL:   {g['total']}")
            print(f"  faces:   {g['faces']}")
            g_faces = g['faces']
    else:
        print("\nGPU no disponible (instala cupy-cudaXX para medir GPU).")

    # CPU
    c = bench_cpu(size, center, reps, iters)
    print("\nCPU:")
    print(f"  Compute: {c['compute']}")
    print(f"  IO:      {c['io']}")
    print(f"  TOTAL:   {c['total']}")
    print(f"  faces:   {c['faces']}")

    print("\nNotas:")
    print("- 100k cubos = 1,200,000 triángulos (~60 MB STL). Requiere algo de tiempo de IO.")
    print("- Usa --no-save para medir solo compute+D→H sin costo de disco.")
    print("- La replicación está vectorizada para minimizar overhead de Python.")

if __name__ == "__main__":
    main()
