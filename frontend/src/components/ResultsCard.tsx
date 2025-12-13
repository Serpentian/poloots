import React from "react";
import type { RootsResponse, BenchmarkResponse, LastProblem } from "../types";

function fmtList(arr?: number[]) {
  if (!arr) return "—";
  if (arr.length === 0) return "∅";
  return arr.join(", ");
}

function formatPoly(coeffs: number[]): string {
  // coeffs: a0..an, строим "x^3 - 2x + 1" (без mod тут, mod добавим отдельно)
  if (!coeffs || coeffs.length === 0) return "0";

  const terms: string[] = [];
  for (let i = coeffs.length - 1; i >= 0; i--) {
    const a = coeffs[i];
    if (!Number.isFinite(a) || a === 0) continue;

    const absA = Math.abs(a);
    const sign = a < 0 ? "-" : "+";

    let term = "";
    if (i === 0) {
      term = `${absA}`;
    } else if (i === 1) {
      if (absA === 1) term = "x";
      else term = `${absA}x`;
    } else {
      if (absA === 1) term = `x^${i}`;
      else term = `${absA}x^${i}`;
    }

    // первый терм без ведущего "+"
    if (terms.length === 0) {
      terms.push(a < 0 ? `-${term}` : term);
    } else {
      terms.push(`${sign} ${term}`);
    }
  }

  return terms.length ? terms.join(" ") : "0";
}

export default function ResultsCard({
  rootsResult,
  benchmarkResult,
  lastProblem
}: {
  rootsResult: RootsResponse | null;
  benchmarkResult: BenchmarkResponse | null;
  lastProblem: LastProblem | null;
}) {
  const cpu = rootsResult?.timings_ms?.sequential;
  const gpu = rootsResult?.timings_ms?.parallel;

  return (
    <div className="card">
      <div className="cardHeader">
        <h2 className="cardTitle">Результаты</h2>
        <div className="pills">
          <span className="pill"><span className="dot dotAccent" /> CPU</span>
          <span className="pill"><span className="dot dotGreen" /> GPU</span>
        </div>
      </div>

      <div className="cardBody">
        {/* Показываем уравнение, если есть последний ввод */}
        <div className="sectionTitle">Что решаем</div>
        {lastProblem ? (
          <div style={{ marginBottom: 14 }}>
            <div className="help">Ищем все x ∈ Fₚ, такие что:</div>
            <div
              className="mono"
              style={{
                marginTop: 8,
                padding: "10px 12px",
                borderRadius: 14,
                border: "1px solid rgba(255,255,255,.12)",
                background: "rgba(0,0,0,.20)",
                fontSize: 14,
                lineHeight: 1.45
              }}
            >
              f(x) = {formatPoly(lastProblem.coeffs)} (mod {lastProblem.p})<br />
              f(x) ≡ 0 (mod {lastProblem.p})
            </div>
          </div>
        ) : (
          <div className="help" style={{ marginBottom: 14 }}>
            Введи коэффициенты и нажми “Вычислить корни” — тут появится уравнение.
          </div>
        )}

        <hr className="hr" />

        <div className="sectionTitle">Корни</div>
        {rootsResult ? (
          <>
            <div className="kv">
              <div className="k">p</div><div className="v"><b>{rootsResult.p}</b></div>
              <div className="k">degree</div><div className="v"><b>{rootsResult.degree}</b></div>
              <div className="k">roots (CPU)</div><div className="v">{fmtList(rootsResult.roots.sequential)}</div>
              <div className="k">roots (GPU)</div><div className="v">{fmtList(rootsResult.roots.parallel)}</div>
              <div className="k">time (ms)</div>
              <div className="v">
                <span className="pill" style={{ marginRight: 8 }}>
                  <span className="dot dotAccent" /> CPU: <b>{cpu ?? "—"}</b>
                </span>
                <span className="pill">
                  <span className="dot dotGreen" /> GPU: <b>{gpu ?? "—"}</b>
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="help">Пока нет данных — запусти “Один запуск”.</div>
        )}

        <hr className="hr" />

        <div className="sectionTitle">Бенчмарк</div>
        {benchmarkResult ? (
          <div className="kv">
            <div className="k">p</div><div className="v"><b>{benchmarkResult.p}</b></div>
            <div className="k">sizes</div><div className="v"><span className="mono">{JSON.stringify(benchmarkResult.sizes)}</span></div>
            <div className="k">заметка</div>
            <div className="v" style={{ color: "rgba(255,255,255,.78)" }}>
              Если GPU быстрее на больших размерах — это ожидаемо. На малых размерах overhead может доминировать.
            </div>
          </div>
        ) : (
          <div className="help">Запусти бенчмарк, чтобы увидеть график.</div>
        )}
      </div>

      <div className="cardFooter">
        <div className="help">
          Никто не поможет.
        </div>
      </div>
    </div>
  );
}
