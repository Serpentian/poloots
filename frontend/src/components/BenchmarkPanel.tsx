import React, { useState } from "react";
import { api } from "../api";
import type { Algorithm, BenchmarkResponse } from "../types";
import TimeChart from "./TimeChart";

function parseSizes(input: string): number[] {
  return input
    .split(/[\s,;]+/g)
    .map((x) => x.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n) && n > 0);
}

export default function BenchmarkPanel({ onResult }: { onResult: (r: BenchmarkResponse) => void }) {
  const [p, setP] = useState<number>(101);
  const [algorithm, setAlgorithm] = useState<Algorithm>("both");
  const [sizesText, setSizesText] = useState<string>("32 64 128 256 512 1024");
  const [seed, setSeed] = useState<number>(123);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localResult, setLocalResult] = useState<BenchmarkResponse | null>(null);

  async function run() {
    const sizes = parseSizes(sizesText);
    setLoading(true);
    setError(null);
    try {
      const res = await api.benchmark({ p, algorithm, sizes, seed });
      setLocalResult(res);
      onResult(res);
    } catch (e: any) {
      setError(e?.message ?? "Ошибка запроса");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <div className="cardHeader">
        <h2 className="cardTitle">Бенчмарк и график</h2>
        <div className="pills">
          <span className="pill"><span className="dot dotWarn" /> time vs size</span>
        </div>
      </div>

      <div className="cardBody">
        <div className="row">
          <div className="field">
            <div className="label">Модуль p</div>
            <input
              className="input"
              type="number"
              value={p}
              onChange={(e) => setP(Number(e.target.value))}
              min={2}
            />
          </div>

          <div className="field">
            <div className="label">Алгоритм</div>
            <select
              className="select"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
            >
                <option value="sequential">Последовательный</option>
                <option value="cpu_parallel">Параллельный CPU</option>
                <option value="gpu_opencl">Параллельный GPU (OpenCL)</option>
                <option value="all">Все три</option>
            </select>
          </div>
        </div>

        <div className="field">
          <div className="label">Размеры входа (список)</div>
          <input
            className="input"
            value={sizesText}
            onChange={(e) => setSizesText(e.target.value)}
          />
          <div className="help">
            Обычно это степень многочлена (или размер пакета тестов) — как реализовано на бэкенде.
          </div>
        </div>

        <div className="field">
          <div className="label">Seed</div>
          <input
            className="input"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </div>

        <div className="actions">
          <button className="btn btnPrimary" onClick={run} disabled={loading}>
            {loading ? "Бенчмарк…" : "Запустить бенчмарк"}
          </button>
        </div>

        {error && <div className="alert alertError">{error}</div>}

        {localResult && (
          <div style={{ marginTop: 14 }}>
            <TimeChart data={localResult} />
          </div>
        )}
      </div>
    </div>
  );
}
