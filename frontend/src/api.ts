import type { RootsRequest, RootsResponse, BenchmarkRequest, BenchmarkResponse } from "./types";

async function postJSON<TReq, TRes>(url: string, body: TReq): Promise<TRes> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${txt || res.statusText}`);
  }
  return (await res.json()) as TRes;
}

export const api = {
  roots: (req: RootsRequest) => postJSON<RootsRequest, RootsResponse>("/api/roots", req),
  benchmark: (req: BenchmarkRequest) => postJSON<BenchmarkRequest, BenchmarkResponse>("/api/benchmark", req)
};
