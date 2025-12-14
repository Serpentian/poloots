import React, { useState } from "react";
import { api } from "../api";
import type { Algorithm, BenchmarkResponse } from "../types";
import TimeChart from "./TimeChart";

/* ---------- utils ---------- */
function parseList(input: string): number[] {
  return input
    .split(/[\s,;]+/g)
    .map((x) => x.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n) && n > 0);
}

export default function BenchmarkPanel({
  onResult
}: {
  onResult: (r: BenchmarkResponse) => void;
}) {
  const [pText, setPText] = useState<string>("101");
  const [degreeText, setDegreeText] = useState<string>("32 64 128 256 512 1024");
  const [algorithm, setAlgorithm] = useState<Algorithm>("all");
  const [seed, setSeed] = useState<number>(123);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localResult, setLocalResult] = useState<BenchmarkResponse | null>(null);

  async function run() {
    const p_values = parseList(pText);
    const degree_values = parseList(degreeText);

    if (p_values.length === 0) {
      setError("Введите хотя бы одно значение p");
      return;
    }
    if (degree_values.length === 0) {
      setError("Введите хотя бы одну степень многочлена");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await api.benchmark({
        p_values,
        degree_values,
        algorithm,
        seed
      });
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
        <h2 className="cardTitle">Бенчмарк и визуализация</h2>
        <div className="pills">
          <span className="pill">
            <span className="dot dotWarn" /> time = f(degree, p)
          </span>
        </div>
      </div>

      <div className="cardBody">
        {/* --- row 1 --- */}
        <div className="row">
          <div className="field">
            <div className="label">Модули p (список)</div>
            <input
              className="input"
              value={pText}
              onChange={(e) => setPText(e.target.value)}
              placeholder="например: 101 503 1009"
            />
            <div className="help">
              Один p → 2D график. Несколько p + степени → 3D.
            </div>
          </div>

          <div className="field">
            <div className="label">Алгоритм</div>
            <select
              className="select"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
            >
              <option value="sequential">Последовательный (CPU)</option>
              <option value="cpu_parallel">Параллельный CPU</option>
              <option value="gpu_opencl">Параллельный GPU (OpenCL)</option>
              <option value="all">Все три</option>
            </select>
          </div>
        </div>

        {/* --- row 2 --- */}
        <div className="field">
          <div className="label">Степени многочлена (degree)</div>
          <input
            className="input"
            value={degreeText}
            onChange={(e) => setDegreeText(e.target.value)}
            placeholder="например: 16 32 64 128"
          />
          <div className="help">
            Одна степень + много p → 2D график.
            Несколько степеней + несколько p → 3D поверхность.
          </div>
        </div>

        {/* --- row 3 --- */}
        <div className="field">
          <div className="label">Seed</div>
          <input
            className="input"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </div>

        {/* --- actions --- */}
        <div className="actions">
          <button
            className="btn btnPrimary"
            onClick={run}
            disabled={loading}
          >
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
