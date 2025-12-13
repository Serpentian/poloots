import React from "react";
import type { RootsResponse, BenchmarkResponse } from "../types";

function fmtList(arr?: number[]) {
  if (!arr) return "—";
  if (arr.length === 0) return "∅";
  return arr.join(", ");
}

export default function ResultsCard({
  rootsResult,
  benchmarkResult
}: {
  rootsResult: RootsResponse | null;
  benchmarkResult: BenchmarkResponse | null;
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
        <div className="sectionTitle">Один многочлен</div>
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
          Для честного сравнения: прогрев CUDA + несколько прогонов и медиана (лучше делать на бэкенде).
        </div>
      </div>
    </div>
  );
}
