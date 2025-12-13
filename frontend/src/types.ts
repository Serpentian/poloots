export type Algorithm = "sequential" | "parallel" | "both";

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

export type BenchmarkRequest = {
  p: number;
  algorithm: Algorithm;      // обычно "both", чтобы сравнить
  sizes: number[];           // “размеры входа”, например степени многочлена или количество тестов
  // Ниже — как именно бэкенд трактует size (степень/кол-во полиномов/и т.п.) — на твоё усмотрение.
  // Главное — чтобы бэкенд вернул времена по каждому size.
  seed?: number;
};

export type BenchmarkResponse = {
  p: number;
  sizes: number[];
  timings_ms: {
    sequential?: number[];
    parallel?: number[];
  };
};
