# cuda.py
from __future__ import annotations
import sys
from dataclasses import dataclass, asdict

try:
    import torch
except Exception:
    torch = None  # PyTorch no instalado

@dataclass
class _Report:
    backend: str
    available: bool
    device: str
    device_count: int
    device_name: str | None
    capability: tuple[int, int] | None
    torch_version: str | None
    python_version: str
    cuda_compiled: bool
    cuda_available: bool
    cuda_version: str | None
    mps_available: bool
    vram_total_gb: float | None
    vram_free_gb: float | None

    def to_dict(self) -> dict:
        return asdict(self)

class cuda:
    @classmethod
    def _detectar(cls, prefer_cuda: bool = True, prefer_mps: bool = True) -> _Report:
        if torch is None:
            return _Report("unavailable", False, "cpu", 0, None, None, None,
                           sys.version.split()[0], False, False, None, False, None, None)

        cuda_compiled = torch.version.cuda is not None
        cuda_available = torch.cuda.is_available() if cuda_compiled else False
        device_count  = torch.cuda.device_count() if cuda_available else 0
        device_name   = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
        capability    = torch.cuda.get_device_capability(0) if cuda_available and device_count > 0 else None

        try:
            mps_available = torch.backends.mps.is_available()
        except Exception:
            mps_available = False

        backend, device_str, available = "cpu", "cpu", False
        if prefer_cuda and cuda_available and device_count > 0:
            backend, device_str, available = "cuda", "cuda:0", True
        elif prefer_mps and mps_available:
            backend, device_str, available = "mps", "mps", True

        vram_free_gb = vram_total_gb = None
        if backend == "cuda":
            try:
                free, total = torch.cuda.mem_get_info()
                vram_free_gb  = round(free/1e9, 2)
                vram_total_gb = round(total/1e9, 2)
            except Exception:
                pass

        return _Report(
            backend=backend,
            available=available,
            device=device_str,
            device_count=device_count,
            device_name=device_name,
            capability=capability,
            torch_version=torch.__version__,
            python_version=sys.version.split()[0],
            cuda_compiled=cuda_compiled,
            cuda_available=cuda_available,
            cuda_version=torch.version.cuda,
            mps_available=mps_available,
            vram_total_gb=vram_total_gb,
            vram_free_gb=vram_free_gb,
        )

    @classmethod
    def probar(cls, prefer_cuda: bool = True, prefer_mps: bool = True) -> _Report:
        rep = cls._detectar(prefer_cuda, prefer_mps)
        print("=== Diagnóstico de aceleración ===")
        print(f"Backend seleccionado : {rep.backend}")
        print(f"Disponible           : {rep.available}")
        print(f"Dispositivo          : {rep.device}")
        print(f"PyTorch              : {rep.torch_version} (Python {rep.python_version})")
        print(f"CUDA compilado       : {rep.cuda_compiled}")
        print(f"CUDA disponible      : {rep.cuda_available}")
        print(f"Versión CUDA (torch) : {rep.cuda_version}")
        print(f"GPUs detectadas      : {rep.device_count}")
        print(f"Nombre GPU[0]        : {rep.device_name}")
        print(f"Compute capability   : {rep.capability}")
        print(f"MPS disponible (Apple): {rep.mps_available}")
        if rep.vram_total_gb is not None:
            print(f"Memoria GPU          : {rep.vram_free_gb} GB libres / {rep.vram_total_gb} GB totales")
        return rep

    @classmethod
    def device(cls, prefer_cuda: bool = True, prefer_mps: bool = True):
        if torch is None:
            import types
            return types.SimpleNamespace(type="cpu", __str__=lambda: "cpu")
        rep = cls._detectar(prefer_cuda, prefer_mps)
        return torch.device(rep.device)

if __name__ == "__main__":
    cuda.probar()
