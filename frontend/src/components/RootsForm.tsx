import React, { useMemo, useState } from "react";
import { api } from "../api";
import type { Algorithm, RootsResponse } from "../types";

function parseCoeffs(input: string): number[] {
  const cleaned = input.trim().replace(/^\[|\]$/g, "");
  if (!cleaned) return [];
  return cleaned
    .split(/[\s,;]+/g)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => Number(x));
}

export default function RootsForm({
  onResult,
  onProblemChange
}: {
  onResult: (r: RootsResponse) => void;
  onProblemChange: (p: number, coeffs: number[]) => void;
}) {
  const [p, setP] = useState<number>(101);
  const [coeffsText, setCoeffsText] = useState<string>("1 0 0 1");
  const [algorithm, setAlgorithm] = useState<Algorithm>("both");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const coeffs = useMemo(() => parseCoeffs(coeffsText), [coeffsText]);
  const degree = Math.max(0, coeffs.length - 1);

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      onProblemChange(p, coeffs);
      const res = await api.roots({ p, coeffs, algorithm });
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
        <h2 className="cardTitle">Один запуск</h2>
        <div className="pills">
          <span className="pill"><span className="dot dotAccent" /> roots</span>
          <span className="pill"><span className="dot dotWarn" /> timing</span>
        </div>
      </div>

      <div className="cardBody">
        <div className="field">
          <div className="label">Модуль p (простое)</div>
          <input
            className="input"
            type="number"
            value={p}
            onChange={(e) => setP(Number(e.target.value))}
            min={2}
          />
        </div>

        <div className="field">
          <div className="label">Коэффициенты (a0 a1 ... an)</div>
          <input
            className="input"
            value={coeffsText}
            onChange={(e) => setCoeffsText(e.target.value)}
            placeholder="например: 1 0 0 1"
          />
          <div className="help">
            Степень: <b>{degree}</b> · coeffs: <span className="mono">{JSON.stringify(coeffs)}</span>
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
            <option value="parallel">Параллельный (GPU)</option>
            <option value="both">Оба (сравнить)</option>
          </select>
        </div>

        <div className="actions">
          <button
            className="btn btnPrimary"
            onClick={submit}
            disabled={loading || coeffs.length === 0 || !Number.isFinite(p)}
          >
            {loading ? "Считаю…" : "Вычислить корни"}
          </button>
        </div>

        {error && <div className="alert alertError">{error}</div>}
      </div>

      <div className="cardFooter">
        <div className="help">
          На малых степенях GPU может проигрывать из-за overhead запуска/копирования — это нормально.
        </div>
      </div>
    </div>
  );
}
