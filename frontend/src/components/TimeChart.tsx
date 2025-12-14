import React, { useState } from "react";
import type { BenchmarkResponse, BenchmarkPoint } from "../types";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
  Plugin
} from "chart.js";
import { Line } from "react-chartjs-2";

import Plot from "react-plotly.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Legend, Tooltip);

/* 🔹 Плагин: белый фон canvas (для 2D) */
const whiteBackground: Plugin = {
  id: "whiteBackground",
  beforeDraw: (chart) => {
    const { ctx, width, height } = chart;
    ctx.save();
    ctx.globalCompositeOperation = "destination-over";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  }
};

type AlgoKey = "sequential" | "cpu_parallel" | "gpu_opencl";
type ViewMode = "points" | "surface" | "both";

const ALGO_LABELS: Record<AlgoKey, string> = {
  sequential: "CPU sequential",
  cpu_parallel: "CPU parallel",
  gpu_opencl: "GPU OpenCL"
};

/* 🎨 единые цвета алгоритмов */
const ALGO_COLORS: Record<AlgoKey, string> = {
  sequential: "rgb(52, 102, 255)",   // blue
  cpu_parallel: "rgb(46, 160, 67)",  // green
  gpu_opencl: "rgb(214, 39, 40)"     // red
};

/* ========================================================= */
/* =================== helpers ============================= */
/* ========================================================= */

function buildSurface(points: BenchmarkPoint[]) {
  const degrees = Array.from(new Set(points.map(p => p.degree))).sort((a, b) => a - b);
  const ps = Array.from(new Set(points.map(p => p.p))).sort((a, b) => a - b);

  const z: number[][] = ps.map(() => Array(degrees.length).fill(NaN));

  for (const pt of points) {
    const i = ps.indexOf(pt.p);
    const j = degrees.indexOf(pt.degree);
    if (i >= 0 && j >= 0) {
      z[i][j] = pt.time_ms;
    }
  }

  return { x: degrees, y: ps, z };
}

/* ========================================================= */
/* =================== component =========================== */
/* ========================================================= */

export default function TimeChart({ data }: { data: BenchmarkResponse }) {
  const [viewMode, setViewMode] = useState<ViewMode>("points");

  const allPoints: BenchmarkPoint[] = Object.values(data.points)
    .flat()
    .filter(Boolean);

  if (allPoints.length === 0) {
    return <div className="chartWrap">Нет данных для отображения</div>;
  }

  const pSet = new Set(allPoints.map(p => p.p));
  const dSet = new Set(allPoints.map(p => p.degree));

  const is2D =
    (pSet.size > 1 && dSet.size === 1) ||
    (pSet.size === 1 && dSet.size > 1);

  const is3D = pSet.size > 1 && dSet.size > 1;

  /* =========================================================
     ======================= 2D ==============================
     ========================================================= */
  if (is2D) {
    const varyingByDegree = dSet.size > 1;
    const labels = [...(varyingByDegree ? dSet : pSet)]
      .sort((a, b) => a - b)
      .map(String);

    const datasets = (Object.keys(data.points) as AlgoKey[]).map((algo) => {
      const pts = data.points[algo];
      if (!pts || pts.length === 0) return null;

      const sorted = [...pts].sort((a, b) =>
        varyingByDegree ? a.degree - b.degree : a.p - b.p
      );

      return {
        label: `${ALGO_LABELS[algo]}, ms`,
        data: sorted.map(p => p.time_ms),
        borderWidth: 2,
        borderColor: ALGO_COLORS[algo]
      };
    }).filter(Boolean);

    return (
      <div className="chartWrap">
        <Line
          plugins={[whiteBackground]}
          data={{ labels, datasets: datasets as any }}
          options={{
            responsive: true,
            plugins: {
              legend: {
                position: "bottom",
                labels: { color: "#111" }
              }
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: varyingByDegree ? "degree" : "p",
                  color: "#000"
                }
              },
              y: {
                title: { display: true, text: "time (ms)", color: "#000" }
              }
            }
          }}
        />
      </div>
    );
  }

  /* =========================================================
     ======================= 3D ==============================
     ========================================================= */
  if (is3D) {
    const traces: any[] = [];

    (Object.keys(data.points) as AlgoKey[]).forEach((algo) => {
      const pts = data.points[algo];
      if (!pts || pts.length === 0) return;

      const color = ALGO_COLORS[algo];

      if (viewMode === "points" || viewMode === "both") {
        traces.push({
          type: "scatter3d",
          mode: "markers",
          name: `${ALGO_LABELS[algo]} (points)`,
          x: pts.map(p => p.degree),
          y: pts.map(p => p.p),
          z: pts.map(p => p.time_ms),
          marker: {
            size: 4,
            color,
            opacity: 0.9
          }
        });
      }

      if (viewMode === "surface" || viewMode === "both") {
        const surf = buildSurface(pts);
        traces.push({
          type: "surface",
          name: `${ALGO_LABELS[algo]} (surface)`,
          x: surf.x,
          y: surf.y,
          z: surf.z,
          surfacecolor: surf.z,
          colorscale: [
            [0, color],
            [1, color]
          ],
          opacity: 0.65,
          showscale: false
        });
      }
    });

    return (
      <div className="chartWrap">
        {/* --- view mode switch --- */}
        <div style={{ marginBottom: 8 }}>
          <select
            className="select"
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as ViewMode)}
          >
            <option value="points">Точки</option>
            <option value="surface">Поверхность</option>
            <option value="both">Точки + поверхность</option>
          </select>
        </div>

        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 560,
            scene: {
              xaxis: { title: "degree" },
              yaxis: { title: "p" },
              zaxis: { title: "time (ms)" }
            },
            legend: {
              orientation: "h",
              y: -0.15
            },
            margin: { l: 0, r: 0, b: 0, t: 30 }
          }}
          style={{ width: "100%" }}
          config={{ displayModeBar: true }}
        />
      </div>
    );
  }

  return <div className="chartWrap">Невозможно определить тип графика</div>;
}

