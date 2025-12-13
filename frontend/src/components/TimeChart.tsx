import React from "react";
import type { BenchmarkResponse } from "../types";
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

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Legend, Tooltip);

/* 🔹 Плагин: белый фон canvas */
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

export default function TimeChart({ data }: { data: BenchmarkResponse }) {
  const labels = data.sizes.map(String);

  const datasets: any[] = [];
  if (data.timings_ms.sequential) {
    datasets.push({
      label: "Sequential (CPU), ms",
      data: data.timings_ms.sequential,
      borderWidth: 2
    });
  }
  if (data.timings_ms.parallel) {
    datasets.push({
      label: "Parallel (GPU), ms",
      data: data.timings_ms.parallel,
      borderWidth: 2
    });
  }

  return (
    <div className="chartWrap">
      <Line
        plugins={[whiteBackground]}
        data={{ labels, datasets }}
        options={{
          responsive: true,
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                color: "#111"   // 🔹 чёрный текст легенды
              }
            },
            tooltip: {
              backgroundColor: "#ffffff",
              titleColor: "#000000",
              bodyColor: "#000000",
              borderColor: "#dddddd",
              borderWidth: 1
            }
          },
          scales: {
            x: {
              title: { display: true, text: "input size", color: "#000" },
              ticks: { color: "#000" },
              grid: { color: "#e5e5e5" }
            },
            y: {
              title: { display: true, text: "ms", color: "#000" },
              ticks: { color: "#000" },
              grid: { color: "#e5e5e5" }
            }
          }
        }}
      />
    </div>
  );
}
