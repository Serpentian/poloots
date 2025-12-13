import React, { useState } from "react";
import RootsForm from "./components/RootsForm";
import ResultsCard from "./components/ResultsCard";
import BenchmarkPanel from "./components/BenchmarkPanel";
import type { RootsResponse, BenchmarkResponse } from "./types";

export default function App() {
  const [rootsResult, setRootsResult] = useState<RootsResponse | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <h1 className="h1">Корни многочлена в поле Fₚ</h1>
          <p className="sub">
            Ввод → вычисление корней (CPU/GPU) → бенчмарк → график времени
          </p>
        </div>

        <div className="pills" aria-label="legend">
          <span className="pill"><span className="dot dotAccent" /> CPU</span>
          <span className="pill"><span className="dot dotGreen" /> GPU</span>
          <span className="pill"><span className="dot dotWarn" /> Benchmark</span>
        </div>
      </div>

      <div className="grid">
        <div style={{ display: "grid", gap: 16 }}>
          <RootsForm onResult={setRootsResult} />
          <BenchmarkPanel onResult={setBenchmarkResult} />
        </div>

        <ResultsCard rootsResult={rootsResult} benchmarkResult={benchmarkResult} />
      </div>
    </div>
  );
}
