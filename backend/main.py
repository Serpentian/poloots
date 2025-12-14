from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

Algorithm = Literal[
    "sequential",
    "cpu_parallel",
    "gpu_opencl",
    "all",
]

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fp_roots")

# ---------------- Optional OpenCL ----------------
OPENCL_AVAILABLE = False
try:
    import pyopencl as cl  # type: ignore
    OPENCL_AVAILABLE = True
except Exception as e:
    logger.warning("pyopencl not available: %s", e)

# ---------------- Optional CPU-parallel ----------------
from numba import njit, prange


# ---------------- Models ----------------
class RootsRequest(BaseModel):
    p: int = Field(..., ge=2)
    coeffs: List[int] = Field(..., min_length=1)
    algorithm: Algorithm


class RootsResponse(BaseModel):
    p: int
    degree: int
    roots: Dict[str, List[int]]
    timings_ms: Dict[str, float]
    parallel_backend: Optional[str] = None


class BenchmarkRequest(BaseModel):
    p_values: Optional[List[int]] = None
    algorithm: Algorithm
    degree_values: Optional[List[int]] = None
    seed: Optional[int] = 123


class BenchmarkPoint(BaseModel):
    p: int
    degree: int
    time_ms: float


class BenchmarkResponse(BaseModel):
    points: Dict[str, List[BenchmarkPoint]]


# ---------------- Utils ----------------
def normalize_coeffs(coeffs: List[int], p: int) -> np.ndarray:
    a = (np.array(coeffs, dtype=np.int64) % p).astype(np.uint32)
    while a.size > 1 and a[-1] == 0:
        a = a[:-1]
    return a


@njit(cache=True)
def horner_eval_modp(coeffs_u32: np.ndarray, x: int, p: int) -> int:
    acc = 0
    for i in range(coeffs_u32.size - 1, -1, -1):
        acc = (acc * x + int(coeffs_u32[i])) % p
    return acc


def roots_sequential(coeffs_u32: np.ndarray, p: int) -> List[int]:
    return [x for x in range(p) if horner_eval_modp(coeffs_u32, x, p) == 0]


@njit(parallel=True, cache=True)
def roots_cpu_parallel_flags(coeffs_u32: np.ndarray, p: int, flags: np.ndarray):
    for x in prange(p):
        acc = 0
        for i in range(coeffs_u32.size - 1, -1, -1):
            acc = (acc * x + int(coeffs_u32[i])) % p
        flags[x] = 1 if acc == 0 else 0


def roots_cpu_parallel(coeffs_u32: np.ndarray, p: int) -> List[int]:
    flags = np.zeros(p, dtype=np.uint8)
    roots_cpu_parallel_flags(coeffs_u32, p, flags)
    return np.nonzero(flags)[0].astype(int).tolist()


def timed(fn, *args):
    t0 = time.perf_counter()
    res = fn(*args)
    return res, (time.perf_counter() - t0) * 1000.0


# ---------------- OpenCL ----------------
OPENCL_KERNEL = r"""
__kernel void find_roots(
    __global const uint* coeffs,
    const uint degree,
    const uint p,
    __global uchar* flags
){
    uint x = get_global_id(0);
    if (x >= p) return;

    uint acc = 0;
    for (int i = (int)degree; i >= 0; --i) {
        ulong t = (ulong)acc * (ulong)x + coeffs[i];
        acc = (uint)(t % p);
    }
    flags[x] = (acc == 0);
}
"""


@dataclass
class OpenCLManager:
    ctx: "cl.Context"
    queue: "cl.CommandQueue"
    program: "cl.Program"
    kernel_find_roots: "cl.Kernel"
    device_name: str

    @staticmethod
    def create() -> "OpenCLManager":
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms")

        dev = platforms[0].get_devices()[0]
        ctx = cl.Context([dev])
        queue = cl.CommandQueue(ctx)
        program = cl.Program(ctx, OPENCL_KERNEL).build()
        kernel = cl.Kernel(program, "find_roots")

        return OpenCLManager(
            ctx=ctx,
            queue=queue,
            program=program,
            kernel_find_roots=kernel,
            device_name=f"{dev.platform.name} / {dev.name}",
        )

    def roots_opencl(self, coeffs_u32: np.ndarray, p: int) -> List[int]:
        mf = cl.mem_flags
        coeffs_u32 = np.ascontiguousarray(coeffs_u32)
        flags = np.empty(p, dtype=np.uint8)

        d_coeffs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=coeffs_u32)
        d_flags = cl.Buffer(self.ctx, mf.WRITE_ONLY, flags.nbytes)

        degree = np.uint32(len(coeffs_u32) - 1)
        self.kernel_find_roots.set_args(
            d_coeffs,
            degree,
            np.uint32(p),
            d_flags,
        )

        cl.enqueue_nd_range_kernel(
            self.queue,
            self.kernel_find_roots,
            (p,),
            None,
        ).wait()

        cl.enqueue_copy(self.queue, flags, d_flags).wait()
        return np.nonzero(flags)[0].tolist()


_opencl_mgr: Optional[OpenCLManager] = None

def get_opencl_mgr() -> Optional[OpenCLManager]:
    global _opencl_mgr
    if not OPENCL_AVAILABLE:
        return None
    if _opencl_mgr is None:
        try:
            _opencl_mgr = OpenCLManager.create()
        except Exception as e:
            logger.warning("OpenCL init failed: %s", e)
            _opencl_mgr = None
    return _opencl_mgr


# ---------------- FastAPI ----------------
app = FastAPI(title="Fp Polynomial Roots", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    t0 = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "req id=%s %s %s -> %d (%.2f ms)",
        rid,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
    )
    response.headers["x-request-id"] = rid
    return response


@app.post("/api/roots", response_model=RootsResponse)
def api_roots(req: RootsRequest):
    p = req.p
    coeffs = normalize_coeffs(req.coeffs, p)

    roots: Dict[str, List[int]] = {}
    timings: Dict[str, float] = {}

    if req.algorithm in ("sequential", "all"):
        r, ms = timed(roots_sequential, coeffs, p)
        roots["sequential"] = r
        timings["sequential"] = ms

    if req.algorithm in ("cpu_parallel", "all"):
        r, ms = timed(roots_cpu_parallel, coeffs, p)
        roots["cpu_parallel"] = r
        timings["cpu_parallel"] = ms

    if req.algorithm in ("gpu_opencl", "all"):
        mgr = get_opencl_mgr()
        if mgr is not None:
            r, ms = timed(mgr.roots_opencl, coeffs, p)
            roots["gpu_opencl"] = r
            timings["gpu_opencl"] = ms
        else:
            roots["gpu_opencl"] = []

    return RootsResponse(
        p=p,
        degree=len(coeffs) - 1,
        roots=roots,
        timings_ms=timings,
        parallel_backend="opencl" if get_opencl_mgr() else "cpu",
    )

def random_poly(degree: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """
    Генерирует случайный многочлен степени `degree` над F_p.
    Коэффициенты: uint32 в [0, p-1], старший коэффициент != 0.
    """
    coeffs = rng.integers(0, p, size=(degree + 1,), dtype=np.uint32)
    if coeffs[-1] == 0:
        coeffs[-1] = np.uint32(rng.integers(1, p))
    return coeffs

@app.post("/api/benchmark", response_model=BenchmarkResponse)
def api_benchmark(req: BenchmarkRequest):
    rng = np.random.default_rng(req.seed)

    p_values = req.p_values
    degree_values = req.degree_values
    alg = req.algorithm

    results: Dict[str, List[BenchmarkPoint]] = {}

    def add_point(key: str, p: int, deg: int, ms: float):
        results.setdefault(key, []).append(
            BenchmarkPoint(p=p, degree=deg, time_ms=ms)
        )

    # warm-up
    warm_poly = random_poly(8, p_values[0], rng)
    _ = roots_sequential(warm_poly, p_values[0])
    _ = roots_cpu_parallel(warm_poly, p_values[0])
    if alg in ("gpu_opencl", "all"):
        mgr = get_opencl_mgr()
        if mgr:
            _ = mgr.roots_opencl(warm_poly, p_values[0])

    for p in p_values:
        for deg in degree_values:
            coeffs = random_poly(deg, p, rng)

            if alg in ("sequential", "all"):
                _, ms = timed(roots_sequential, coeffs, p)
                add_point("sequential", p, deg, ms)

            if alg in ("cpu_parallel", "all"):
                _, ms = timed(roots_cpu_parallel, coeffs, p)
                add_point("cpu_parallel", p, deg, ms)

            if alg in ("gpu_opencl", "all"):
                mgr = get_opencl_mgr()
                if mgr:
                    _, ms = timed(mgr.roots_opencl, coeffs, p)
                    add_point("gpu_opencl", p, deg, ms)
                else:
                    add_point("gpu_opencl", p, deg, float("nan"))

    return BenchmarkResponse(points=results)

# Run:
# LOG_LEVEL=DEBUG uvicorn main:app --host 0.0.0.0 --port 8000
