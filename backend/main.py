from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

Algorithm = Literal["sequential", "parallel", "both"]

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
    OPENCL_AVAILABLE = False
    logger.warning("pyopencl not available: %s", e)

# ---------------- Optional CPU-parallel fallback ----------------
from numba import njit, prange


# ---------------- Models ----------------
class RootsRequest(BaseModel):
    p: int = Field(..., ge=2)
    coeffs: List[int] = Field(..., min_length=1)  # a0..an
    algorithm: Algorithm


class RootsResponse(BaseModel):
    p: int
    degree: int
    roots: Dict[str, List[int]]
    timings_ms: Optional[Dict[str, float]] = None
    parallel_backend: Optional[str] = None  # "opencl:<device>" | "cpu-parallel"


class BenchmarkRequest(BaseModel):
    p: int = Field(..., ge=2)
    algorithm: Algorithm
    sizes: List[int] = Field(..., min_length=1)  # degree list
    seed: Optional[int] = 123


class BenchmarkResponse(BaseModel):
    p: int
    sizes: List[int]
    timings_ms: Dict[str, List[float]]
    parallel_backend: str


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
    roots: List[int] = []
    for x in range(p):
        if horner_eval_modp(coeffs_u32, x, p) == 0:
            roots.append(x)
    return roots


@njit(parallel=True, cache=True)
def roots_cpu_parallel_flags(coeffs_u32: np.ndarray, p: int, out_flags: np.ndarray):
    for x in prange(p):
        acc = 0
        for i in range(coeffs_u32.size - 1, -1, -1):
            acc = (acc * x + int(coeffs_u32[i])) % p
        out_flags[x] = 1 if acc == 0 else 0


def roots_cpu_parallel(coeffs_u32: np.ndarray, p: int) -> List[int]:
    flags = np.zeros(p, dtype=np.uint8)
    roots_cpu_parallel_flags(coeffs_u32, p, flags)
    return np.nonzero(flags)[0].astype(int).tolist()


def timed(fn, *args):
    t0 = time.perf_counter()
    res = fn(*args)
    t1 = time.perf_counter()
    return res, (t1 - t0) * 1000.0


# ---------------- OpenCL engine ----------------
OPENCL_KERNEL = r"""
__kernel void find_roots(
    __global const uint* coeffs,
    const uint degree,
    const uint p,
    __global uchar* flags
){
    uint x = (uint)get_global_id(0);
    if (x >= p) return;

    uint acc = 0u;
    // Horner: i = degree..0
    for (int i = (int)degree; i >= 0; --i) {
        // acc = (acc*x + coeffs[i]) % p with 64-bit intermediate
        ulong t = (ulong)acc * (ulong)x + (ulong)coeffs[i];
        acc = (uint)(t % (ulong)p);
    }
    flags[x] = (acc == 0u) ? (uchar)1 : (uchar)0;
}
"""


@dataclass
class OpenCLManager:
    ctx: "cl.Context"
    queue: "cl.CommandQueue"
    program: "cl.Program"
    device_name: str

    @staticmethod
    def create() -> "OpenCLManager":
        if not OPENCL_AVAILABLE:
            raise RuntimeError("pyopencl не установлен")

        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("OpenCL платформы не найдены (cl.get_platforms пусто)")

        logger.info("OpenCL platforms found: %d", len(platforms))

        # Prefer GPU device if possible, else any device
        chosen_dev = None
        chosen_kind = "unknown"

        for plat in platforms:
            try:
                gpus = plat.get_devices(device_type=cl.device_type.GPU)
            except Exception:
                gpus = []
            if gpus:
                chosen_dev = gpus[0]
                chosen_kind = "GPU"
                break

        if chosen_dev is None:
            for plat in platforms:
                try:
                    devs = plat.get_devices()
                except Exception:
                    devs = []
                if devs:
                    chosen_dev = devs[0]
                    chosen_kind = "ANY"
                    break

        if chosen_dev is None:
            raise RuntimeError("OpenCL устройства не найдены")

        ctx = cl.Context(devices=[chosen_dev])
        queue = cl.CommandQueue(ctx)

        t0 = time.perf_counter()
        program = cl.Program(ctx, OPENCL_KERNEL).build()
        build_ms = (time.perf_counter() - t0) * 1000.0

        name = f"{chosen_dev.platform.name.strip()} / {chosen_dev.name.strip()}"
        logger.info("OpenCL selected device (%s): %s", chosen_kind, name)
        logger.info("OpenCL program build time: %.2f ms", build_ms)

        return OpenCLManager(ctx=ctx, queue=queue, program=program, device_name=name)

    def roots_opencl(self, coeffs_u32: np.ndarray, p: int) -> List[int]:
        mf = cl.mem_flags
        coeffs_u32 = np.ascontiguousarray(coeffs_u32, dtype=np.uint32)
        degree = np.uint32(coeffs_u32.size - 1)
        p_u32 = np.uint32(p)

        d_coeffs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=coeffs_u32)
        d_flags = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=p * np.dtype(np.uint8).itemsize)

        local = 256
        global_size = ((p + local - 1) // local) * local

        logger.debug("OpenCL launch: p=%d degree=%d global=%d local=%d", p, int(degree), global_size, local)

        evt = self.program.find_roots(
            self.queue,
            (global_size,),
            (local,),
            d_coeffs,
            degree,
            p_u32,
            d_flags,
        )
        evt.wait()

        flags = np.empty(p, dtype=np.uint8)
        cl.enqueue_copy(self.queue, flags, d_flags).wait()

        return np.nonzero(flags)[0].astype(int).tolist()


_opencl_mgr: Optional[OpenCLManager] = None
_opencl_last_error: Optional[str] = None


def get_opencl_mgr() -> Optional[OpenCLManager]:
    global _opencl_mgr, _opencl_last_error
    if not OPENCL_AVAILABLE:
        return None
    if _opencl_mgr is None:
        try:
            logger.info("Initializing OpenCL manager...")
            _opencl_mgr = OpenCLManager.create()
            _opencl_last_error = None
        except Exception as e:
            _opencl_mgr = None
            _opencl_last_error = str(e)
            logger.warning("OpenCL init failed, fallback to cpu-parallel. Reason: %s", e)
    return _opencl_mgr


def pick_parallel_backend() -> str:
    mgr = get_opencl_mgr()
    if mgr is not None:
        return f"opencl:{mgr.device_name}"
    return "cpu-parallel"


def roots_parallel(coeffs_u32: np.ndarray, p: int) -> List[int]:
    mgr = get_opencl_mgr()
    if mgr is not None:
        return mgr.roots_opencl(coeffs_u32, p)
    return roots_cpu_parallel(coeffs_u32, p)


# ---------------- Benchmark poly generator ----------------
def random_poly(degree: int, p: int, rng: np.random.Generator) -> np.ndarray:
    coeffs = rng.integers(0, p, size=(degree + 1,), dtype=np.uint32)
    if coeffs[-1] == 0:
        coeffs[-1] = np.uint32(rng.integers(1, p))
    return coeffs


# ---------------- FastAPI ----------------
app = FastAPI(title="Fp Polynomial Roots (OpenCL)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "req id=%s %s %s -> %d (%.2f ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            dt_ms,
        )
        response.headers["x-request-id"] = request_id
        return response
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.exception(
            "req id=%s %s %s -> EXCEPTION (%.2f ms): %s",
            request_id,
            request.method,
            request.url.path,
            dt_ms,
            e,
        )
        raise


@app.get("/api/health")
def health():
    backend = pick_parallel_backend()
    logger.debug("health: opencl_available=%s backend=%s", OPENCL_AVAILABLE, backend)
    payload = {
        "ok": True,
        "opencl_available": OPENCL_AVAILABLE,
        "parallel_backend": backend,
    }
    if backend == "cpu-parallel" and _opencl_last_error:
        payload["opencl_error"] = _opencl_last_error
    return payload


@app.post("/api/roots", response_model=RootsResponse)
def api_roots(req: RootsRequest):
    p = int(req.p)
    if p < 2:
        logger.warning("roots: invalid p=%s", req.p)
        raise HTTPException(400, "p должно быть >= 2")

    coeffs_u32 = normalize_coeffs(req.coeffs, p)
    degree = int(coeffs_u32.size - 1)

    logger.info(
        "roots: p=%d degree=%d alg=%s coeffs_len=%d backend=%s",
        p, degree, req.algorithm, len(req.coeffs), pick_parallel_backend()
    )

    roots: Dict[str, List[int]] = {}
    timings: Dict[str, float] = {}

    if req.algorithm in ("sequential", "both"):
        r, ms = timed(roots_sequential, coeffs_u32, p)
        roots["sequential"] = r
        timings["sequential"] = ms
        logger.info("roots: sequential done (%.2f ms), count=%d", ms, len(r))

    if req.algorithm in ("parallel", "both"):
        r, ms = timed(roots_parallel, coeffs_u32, p)
        roots["parallel"] = r
        timings["parallel"] = ms
        logger.info("roots: parallel done (%.2f ms), count=%d backend=%s", ms, len(r), pick_parallel_backend())

    return RootsResponse(
        p=p,
        degree=degree,
        roots=roots,
        timings_ms=timings,
        parallel_backend=pick_parallel_backend(),
    )


@app.post("/api/benchmark", response_model=BenchmarkResponse)
def api_benchmark(req: BenchmarkRequest):
    p = int(req.p)
    sizes = [int(s) for s in req.sizes]
    if any(s <= 0 for s in sizes):
        logger.warning("benchmark: invalid sizes=%s", req.sizes)
        raise HTTPException(400, "Все sizes должны быть > 0")

    rng = np.random.default_rng(req.seed if req.seed is not None else 123)
    backend = pick_parallel_backend()

    logger.info(
        "benchmark: p=%d alg=%s sizes=%s seed=%s backend=%s",
        p, req.algorithm, sizes, req.seed, backend
    )

    # Warm-up
    warm_poly = random_poly(16, p, rng)
    _ = roots_sequential(warm_poly, p)
    if req.algorithm in ("parallel", "both"):
        _ = roots_parallel(warm_poly, p)

    seq_times: List[float] = []
    par_times: List[float] = []

    for deg in sizes:
        coeffs = random_poly(deg, p, rng)

        if req.algorithm in ("sequential", "both"):
            _, ms = timed(roots_sequential, coeffs, p)
            seq_times.append(ms)
            logger.debug("benchmark: deg=%d sequential=%.2f ms", deg, ms)

        if req.algorithm in ("parallel", "both"):
            _, ms = timed(roots_parallel, coeffs, p)
            par_times.append(ms)
            logger.debug("benchmark: deg=%d parallel=%.2f ms backend=%s", deg, ms, pick_parallel_backend())

    timings: Dict[str, List[float]] = {}
    if req.algorithm in ("sequential", "both"):
        timings["sequential"] = seq_times
    if req.algorithm in ("parallel", "both"):
        timings["parallel"] = par_times

    logger.info(
        "benchmark done: sizes=%d seq=%s par=%s backend=%s",
        len(sizes),
        "yes" if "sequential" in timings else "no",
        "yes" if "parallel" in timings else "no",
        pick_parallel_backend(),
    )

    return BenchmarkResponse(p=p, sizes=sizes, timings_ms=timings, parallel_backend=backend)


# Run:
# LOG_LEVEL=DEBUG uvicorn main:app --host 0.0.0.0 --port 8000
# uvicorn main:app --host 0.0.0.0 --port 8000
