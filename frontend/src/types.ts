export type Algorithm = "sequential" | "cpu_parallel" | "gpu_opencl" | "all";

export type RootsRequest = {
  p: number;                 // модуль
  coeffs: number[];          // коэффициенты [a0, a1, ..., an]
  algorithm: Algorithm;
};

export type RootsResponse = {
  p: number;
  degree: number;
  roots: {
    sequential?: number[];
    parallel?: number[];
  };
  timings_ms?: {
    sequential?: number;
    parallel?: number;
  };
};

export type BenchmarkPoint = {
  p: number;
  degree: number;
  time_ms: number;
};

export type BenchmarkResponse = {
  points: {
    sequential?: BenchmarkPoint[];
    cpu_parallel?: BenchmarkPoint[];
    gpu_opencl?: BenchmarkPoint[];
  };
};

export type BenchmarkRequest = {
  p_values: number[];
  degree_values: number[];
  algorithm: Algorithm;
  seed?: number;
};

export type LastProblem = {
  p: number;
  coeffs: number[]; // a0..an
};
