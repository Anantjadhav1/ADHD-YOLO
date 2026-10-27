import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDown, ArrowRight, BarChart3, Check, FileText, FlaskConical,
  Info, RotateCcw, ShieldAlert, ShieldCheck, Upload, X,
} from "lucide-react";

// In development the page runs on Vite's port and talks to uvicorn on :8000; once built,
// the backend serves the page itself at /app/, so the API is the same origin.
const API = import.meta.env.DEV ? "http://localhost:8000" : "";

const STEPS = [
  { id: 1, label: "Upload" },
  { id: 2, label: "Generate" },
  { id: 3, label: "Results" },
  { id: 4, label: "Validation" },
];

// The backend reports no progress, so these are the stages it runs, in order, with
// typical start times for one EOEC recording on a laptop CPU. Only the response marks
// the work done.
const PIPELINE = [
  { label: "Filter", detail: "0.5–50 Hz, 50 Hz notch", at: 0 },
  { label: "Clean", detail: "ICA: eye and muscle", at: 3 },
  { label: "Segment", detail: "1.5 s epochs", at: 9 },
  { label: "Transform", detail: "Wavelet scalograms", at: 11 },
  { label: "Classify", detail: "YOLOv8n-cls", at: 13 },
  { label: "Explain", detail: "Grad-CAM, scalp maps", at: 18 },
];

const BIOMARKERS = [
  { key: "tbr", label: "Theta / beta ratio", hint: "F3 · F4 · Fz", d: 2 },
  { key: "iaf", label: "Alpha peak", hint: "Hz · O1 / O2", d: 1 },
  { key: "aperiodic_exponent", label: "Aperiodic slope", hint: "1/f exponent", d: 2 },
  { key: "frontal_alpha_asym", label: "Alpha asymmetry", hint: "F4 versus F3", d: 2 },
];

const BANDS = [["delta", "δ Delta"], ["theta", "θ Theta"], ["alpha", "α Alpha"], ["beta", "β Beta"], ["gamma", "γ Gamma"]];
const CONDITION = { EC: "Eyes closed", EO: "Eyes open", VCPT: "Task" };
const COLORS = { adhd: "#a78bfa", control: "#4b5675", self: "#f5d0fe", ec: "#8b5cf6", eo: "#c4b5fd" };

// 10-20 electrode positions on the unit head circle, nose up (azimuthal projection).
const POS = {
  Fp1: [-0.25, 0.76], Fp2: [0.25, 0.76], F7: [-0.65, 0.47], F3: [-0.31, 0.38], Fz: [0, 0.36],
  F4: [0.31, 0.38], F8: [0.65, 0.47], T3: [-0.8, 0], C3: [-0.38, 0], Cz: [0, 0], C4: [0.38, 0],
  T4: [0.8, 0], T5: [-0.65, -0.47], P3: [-0.31, -0.38], Pz: [0, -0.36], P4: [0.31, -0.38],
  T6: [0.65, -0.47], O1: [-0.25, -0.76], O2: [0.25, -0.76],
};

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v) => `${(v * 100).toFixed(0)}%`;
const png = (b64) => `data:image/png;base64,${b64}`;

function useCountUp(target, ms = 1200) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf; let t0;
    const tick = (t) => {
      t0 ??= t;
      const k = Math.min(1, (t - t0) / ms);
      setV(target * (1 - (1 - k) ** 3));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export default function App() {
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState({ eoec: null, vcpt: null });
  const [run, setRun] = useState({ status: "idle", result: null, error: null, started: 0 });
  const [elapsed, setElapsed] = useState(0);
  const [reference, setReference] = useState([]);
  const [study, setStudy] = useState([]);
  const [online, setOnline] = useState(null);

  useEffect(() => {
    fetch(`${API}/health`).then((r) => setOnline(r.ok)).catch(() => setOnline(false));
    fetch(`${API}/reference`).then((r) => r.json()).then(setReference).catch(() => {});
    fetch(`${API}/study`).then((r) => r.json()).then(setStudy).catch(() => {});
  }, []);

  useEffect(() => { window.scrollTo({ top: 0, behavior: "smooth" }); }, [step]);

  useEffect(() => {
    if (run.status !== "running") return undefined;
    const t = setInterval(() => setElapsed((Date.now() - run.started) / 1000), 200);
    return () => clearInterval(t);
  }, [run.status, run.started]);

  async function analyse() {
    const fd = new FormData();
    fd.append("eoec_file", files.eoec);
    if (files.vcpt) fd.append("vcpt_file", files.vcpt);
    setRun({ status: "running", result: null, error: null, started: Date.now() });
    setElapsed(0);
    setStep(2);
    try {
      const res = await fetch(`${API}/predict`, { method: "POST", body: fd });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `The server returned ${res.status}.`);
      setRun((r) => ({ ...r, status: "done", result: body }));
    } catch (e) {
      const msg = e.message === "Failed to fetch"
        ? "The analysis server cannot be reached. Start it with uvicorn and try again."
        : e.message;
      setRun((r) => ({ ...r, status: "error", error: msg }));
    }
  }

  function reset() {
    setFiles({ eoec: null, vcpt: null });
    setRun({ status: "idle", result: null, error: null, started: 0 });
    setStep(1);
  }

  const subjectId = files.eoec ? files.eoec.name.split("-")[0].toUpperCase() : null;
  const reachable = { 1: run.status !== "running", 2: run.status !== "idle", 3: run.status === "done", 4: true };
  const h1a = study.find((r) => r.hypothesis === "H1a");

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="dotgrid pointer-events-none absolute inset-x-0 top-0 h-[900px] [mask-image:linear-gradient(to_bottom,black,transparent)]" />
      <div className="pointer-events-none absolute left-1/2 top-[-260px] h-[900px] w-[1300px] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgba(124,58,237,0.42),rgba(124,58,237,0.12)_55%,transparent)]" />

      <div className="relative">
        <div className="border-b border-white/5 bg-black/40 py-2 text-center text-[12px] text-violet-200/70">
          Research prototype · every analysis runs on this machine · final results pre-registered on GitHub
        </div>
        <NavBar step={step} reachable={reachable} onGo={setStep} online={online}
                canReset={run.status !== "idle" && run.status !== "running"} onReset={reset} />
        <main className="mx-auto max-w-[1200px] px-6 pb-24 lg:px-10">
          <div key={step}>
            {step === 1 && (
              <UploadStage files={files} setFiles={setFiles} onRun={analyse} online={online}
                           h1a={h1a} reference={reference} onStudy={() => setStep(4)} />
            )}
            {step === 2 && (
              <GenerateStage run={run} elapsed={elapsed} subjectId={subjectId}
                             onNext={() => setStep(3)} onBack={reset} />
            )}
            {step === 3 && run.result && (
              <ResultsStage result={run.result} reference={reference} subjectId={subjectId}
                            onStudy={() => setStep(4)} />
            )}
            {step === 4 && <ValidationStage study={study} />}
          </div>
        </main>
        <footer className="border-t border-white/5">
          <div className="mx-auto flex max-w-[1200px] flex-wrap justify-between gap-3 px-6 py-7 text-xs text-violet-200/40 lg:px-10">
            <span>ADHD-YOLO · EEG classification of ADHD in children · thesis research prototype</span>
            <span>A decision-support research tool, not a diagnostic device.</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

/* ================================================================ frame */

function LogoMark({ size = 30 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#c4b5fd" /><stop offset="1" stopColor="#7c3aed" /></linearGradient>
      </defs>
      <circle cx="16" cy="16" r="14" fill="none" stroke="url(#lg)" strokeWidth="2" />
      <path d="M5 17 L10 17 L12.5 11 L16 23 L19 9 L21.5 17 L27 17" fill="none" stroke="#f5f3ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function NavBar({ step, reachable, onGo, online, canReset, onReset }) {
  return (
    <nav className="mx-auto grid max-w-[1200px] grid-cols-[1fr_auto_1fr] items-center gap-4 px-6 py-5 lg:px-10">
      <div className="flex flex-wrap items-center gap-x-2 text-[13px]">
        {STEPS.map((s, i) => (
          <span key={s.id} className="flex items-center gap-2">
            {i > 0 && <span className="text-white/20">/</span>}
            <button type="button" disabled={!reachable[s.id]} onClick={() => onGo(s.id)}
              className={`transition disabled:cursor-not-allowed disabled:text-white/20 ${s.id === step ? "font-medium text-white" : "text-violet-200/55 hover:text-white"}`}>
              {s.label}
            </button>
          </span>
        ))}
      </div>
      <div className="flex items-center gap-2.5">
        <LogoMark />
        <span className="text-[19px] font-semibold tracking-tight text-white">ADHD-YOLO</span>
      </div>
      <div className="flex items-center justify-end gap-2.5">
        <span className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-[12px] text-violet-100/70 sm:flex">
          <span className={`h-1.5 w-1.5 rounded-full ${online === null ? "bg-white/30" : online ? "bg-emerald-400 shadow-[0_0_10px_2px_rgba(52,211,153,0.7)]" : "bg-rose-500"}`} />
          {online === null ? "Connecting" : online ? "Server online" : "Server offline"}
        </span>
        {canReset ? (
          <button type="button" onClick={onReset}
            className="flex items-center gap-2 rounded-lg bg-white px-3.5 py-2 text-[13px] font-medium text-[#120a24] transition hover:bg-violet-100">
            <RotateCcw size={13} /> New analysis
          </button>
        ) : (
          <button type="button" onClick={() => onGo(4)}
            className="rounded-lg bg-white px-3.5 py-2 text-[13px] font-medium text-[#120a24] transition hover:bg-violet-100">
            View the study
          </button>
        )}
      </div>
    </nav>
  );
}

function Title({ children, className = "" }) {
  return <h1 className={`text-balance font-normal leading-[1.04] tracking-[-0.025em] text-white ${className}`}>{children}</h1>;
}
function Em({ children }) {
  return <em className="font-serif font-normal italic tracking-normal text-violet-200">{children}</em>;
}

function PageHead({ eyebrow, title, children, right }) {
  return (
    <div className="fade-up flex flex-wrap items-end justify-between gap-6 pb-10 pt-12">
      <div className="max-w-2xl">
        <div className="text-[12px] font-medium uppercase tracking-[0.22em] text-violet-300/80">{eyebrow}</div>
        <Title className="mt-4 text-[46px]">{title}</Title>
        {children && <p className="mt-4 text-[15px] leading-relaxed text-violet-100/60">{children}</p>}
      </div>
      {right}
    </div>
  );
}

function Glass({ className = "", style, strong, children }) {
  return <div style={style} className={`${strong ? "glass-strong" : "glass"} rounded-[22px] ${className}`}>{children}</div>;
}

function CardHead({ icon: Icon, title, hint, right }) {
  return (
    <div className="mb-6 flex items-start justify-between gap-4">
      <div className="flex items-start gap-3">
        {Icon && <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-violet-500/15 text-violet-300 ring-1 ring-violet-400/20"><Icon size={16} /></span>}
        <div>
          <div className="text-[15px] font-medium text-white">{title}</div>
          {hint && <div className="mt-0.5 text-xs leading-relaxed text-violet-100/50">{hint}</div>}
        </div>
      </div>
      {right}
    </div>
  );
}

function Chip({ children, tone }) {
  const tones = {
    ok: "bg-emerald-400/10 text-emerald-300 ring-emerald-400/25",
    warn: "bg-amber-400/10 text-amber-200 ring-amber-300/25",
    bad: "bg-rose-400/10 text-rose-200 ring-rose-300/25",
    violet: "bg-violet-400/15 text-violet-100 ring-violet-300/30",
    undefined: "bg-white/[0.05] text-violet-100/70 ring-white/10",
  };
  return <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-medium ring-1 ${tones[tone]}`}>{children}</span>;
}

function PrimaryButton({ children, onClick, disabled, className = "" }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled}
      className={`group inline-flex items-center justify-center gap-2 rounded-xl bg-white px-6 py-3.5 text-[14px] font-medium text-[#120a24] shadow-[0_10px_40px_-10px_rgba(167,139,250,0.65)] transition-all duration-300 hover:-translate-y-0.5 hover:bg-violet-50 disabled:translate-y-0 disabled:cursor-not-allowed disabled:bg-white/15 disabled:text-white/40 disabled:shadow-none ${className}`}>
      {children}
    </button>
  );
}

function GhostButton({ children, onClick }) {
  return (
    <button type="button" onClick={onClick}
      className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/15 bg-white/[0.06] px-6 py-3.5 text-[14px] font-medium text-white backdrop-blur transition hover:bg-white/[0.12]">
      {children}
    </button>
  );
}

function Toggle({ value, onChange, options }) {
  return (
    <div className="inline-flex rounded-xl bg-white/[0.05] p-1 ring-1 ring-white/10">
      {options.map(([v, label]) => (
        <button key={v} type="button" onClick={() => onChange(v)}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${value === v ? "bg-white text-[#120a24]" : "text-violet-100/60 hover:text-white"}`}>
          {label}
        </button>
      ))}
    </div>
  );
}

/* ============================================================ head map */

const STOPS = [[0, [18, 10, 36]], [0.25, [46, 16, 101]], [0.5, [109, 40, 217]], [0.75, [168, 85, 247]], [0.9, [232, 121, 249]], [1, [253, 244, 255]]];
function cmap(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i += 1) {
    if (x <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1]; const [b, cb] = STOPS[i]; const k = (x - a) / (b - a);
      return ca.map((c, j) => Math.round(c + (cb[j] - c) * k));
    }
  }
  return STOPS[STOPS.length - 1][1];
}

// Scalp map by inverse-distance interpolation between the 19 electrodes; geometry in head
// units (radius 1), drawn on a canvas under an SVG of the head outline and electrodes.
function HeadMap({ values, size = 440, rays = false, labels = true, pulse = false, legend }) {
  const canvas = useRef(null);
  const pts = useMemo(() => Object.entries(values || {}).filter(([k, v]) => POS[k] && v !== null && v !== undefined), [values]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !pts.length) return;
    const N = 240;
    c.width = N; c.height = N;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(N, N);
    const vals = pts.map((p) => p[1]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    for (let y = 0; y < N; y += 1) {
      for (let x = 0; x < N; x += 1) {
        const u = ((x + 0.5) / N) * 2 - 1, v = 1 - ((y + 0.5) / N) * 2, r = Math.hypot(u, v);
        const i = (y * N + x) * 4;
        if (r > 1) { img.data[i + 3] = 0; continue; }
        let num = 0, den = 0;
        for (const [k, val] of pts) {
          const d2 = (u - POS[k][0]) ** 2 + (v - POS[k][1]) ** 2 + 1e-4;
          const w = 1 / (d2 * d2);
          num += w * val; den += w;
        }
        const [R, G, B] = cmap((num / den - lo) / (hi - lo || 1));
        img.data[i] = R; img.data[i + 1] = G; img.data[i + 2] = B;
        img.data[i + 3] = Math.round(255 * Math.min(1, (1 - r) * 14));
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [pts]);

  const S = 1.5; // svg half-extent in head units
  const px = (u) => ((u + S) / (2 * S)) * 100;
  const box = { left: `${px(-1)}%`, top: `${px(-1)}%`, width: `${px(1) - px(-1)}%`, height: `${px(1) - px(-1)}%` };
  return (
    <div className="relative mx-auto" style={{ width: size, height: size }}>
      {rays && (
        <svg viewBox={`${-S} ${-S} ${2 * S} ${2 * S}`} className="spin-very-slow absolute inset-0" aria-hidden="true">
          {Array.from({ length: 144 }, (_, i) => {
            const a = (i / 144) * 2 * Math.PI, r1 = 1.2, r2 = r1 + (i % 6 === 0 ? 0.13 : 0.06);
            return <line key={i} x1={r1 * Math.cos(a)} y1={r1 * Math.sin(a)} x2={r2 * Math.cos(a)} y2={r2 * Math.sin(a)}
                         stroke="#c4b5fd" strokeOpacity={i % 6 === 0 ? 0.45 : 0.18} strokeWidth="0.008" />;
          })}
          <circle r="1.42" fill="none" stroke="#c4b5fd" strokeOpacity="0.08" strokeWidth="0.006" />
        </svg>
      )}
      <div className="absolute rounded-full shadow-[0_0_120px_10px_rgba(139,92,246,0.35)]" style={box} />
      <canvas ref={canvas} className="absolute rounded-full" style={box} />
      <svg viewBox={`${-S} ${-S} ${2 * S} ${2 * S}`} className="absolute inset-0" role="img" aria-label="Scalp map">
        <path d="M -0.11 -0.99 L 0 -1.13 L 0.11 -0.99" fill="none" stroke="#ddd6fe" strokeOpacity="0.7" strokeWidth="0.018" strokeLinejoin="round" />
        <path d="M -1 -0.12 C -1.08 -0.1 -1.09 0.1 -1 0.12" fill="none" stroke="#ddd6fe" strokeOpacity="0.7" strokeWidth="0.018" />
        <path d="M 1 -0.12 C 1.08 -0.1 1.09 0.1 1 0.12" fill="none" stroke="#ddd6fe" strokeOpacity="0.7" strokeWidth="0.018" />
        <circle r="1" fill="none" stroke="#ddd6fe" strokeOpacity="0.7" strokeWidth="0.018" />
        <circle r="0.8" fill="none" stroke="#ffffff" strokeOpacity="0.1" strokeWidth="0.006" strokeDasharray="0.02 0.03" />
        {Object.entries(POS).map(([k, [u, v]], i) => (
          <g key={k}>
            {pulse && <circle cx={u} cy={-v} r="0.035" fill="#f5d0fe" className="ping-soft" style={{ animationDelay: `${(i % 7) * 0.37}s` }} />}
            <circle cx={u} cy={-v} r="0.03" fill="#faf5ff" stroke="#2e1065" strokeWidth="0.012" />
            {labels && <text x={u} y={-v + 0.1} fontSize="0.068" fill="#f5f3ff" stroke="#160c2c" strokeOpacity="0.85" strokeWidth="0.022" paintOrder="stroke" textAnchor="middle" fontFamily="Inter Variable, sans-serif" fontWeight="500">{k}</text>}
          </g>
        ))}
      </svg>
      {legend && (
        <div className="absolute inset-x-0 -bottom-2 mx-auto flex w-56 items-center gap-2 text-[10px] text-violet-100/50">
          <span>low</span>
          <span className="h-1.5 flex-1 rounded-full" style={{ background: `linear-gradient(90deg, ${STOPS.map(([t, c]) => `rgb(${c.join(",")}) ${t * 100}%`).join(",")})` }} />
          <span>high</span>
        </div>
      )}
    </div>
  );
}

/* =========================================================== 1. upload */

function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Landing-page artwork, decorative only: concentric wave rings built from integer
// harmonics (so each closes on itself), turning slowly in alternate directions.
const ORB_RINGS = Array.from({ length: 16 }, (_, i) => {
  const rnd = mulberry32(11 + i * 7);
  const R = 0.34 + i * 0.043;
  const comps = [[1, 3], [0.55, 7], [0.25, 13], [0.1, 23]].map(([a, f]) => ({ a, f: f + Math.floor(rnd() * 3), ph: rnd() * Math.PI * 2 }));
  const amp = 0.006 + i * 0.0011;
  const at = (t) => R + amp * comps.reduce((s, c) => s + c.a * Math.sin(c.f * t + c.ph), 0);
  let d = "";
  for (let k = 0; k <= 360; k += 1) {
    const t = (k / 360) * 2 * Math.PI, r = at(t);
    d += `${k ? "L" : "M"}${(r * Math.cos(t)).toFixed(4)} ${(r * Math.sin(t)).toFixed(4)}`;
  }
  const t0 = rnd() * 2 * Math.PI;
  return { d: `${d}Z`, dot: [at(t0) * Math.cos(t0), at(t0) * Math.sin(t0)] };
});

function NeuralOrb({ size = 540 }) {
  return (
    <div className="relative mx-auto" style={{ width: size, height: size }} aria-hidden="true">
      <svg viewBox="-1.5 -1.5 3 3" className="spin-very-slow absolute inset-0">
        {Array.from({ length: 144 }, (_, i) => {
          const a = (i / 144) * 2 * Math.PI, r1 = 1.2, r2 = r1 + (i % 6 === 0 ? 0.13 : 0.06);
          return <line key={i} x1={r1 * Math.cos(a)} y1={r1 * Math.sin(a)} x2={r2 * Math.cos(a)} y2={r2 * Math.sin(a)}
                       stroke="#c4b5fd" strokeOpacity={i % 6 === 0 ? 0.45 : 0.18} strokeWidth="0.008" />;
        })}
        <circle r="1.42" fill="none" stroke="#c4b5fd" strokeOpacity="0.08" strokeWidth="0.006" />
      </svg>
      <div className="absolute inset-[20%] rounded-full bg-[radial-gradient(closest-side,rgba(167,139,250,0.5),rgba(91,33,182,0.22)_55%,transparent)] blur-2xl" />
      <svg viewBox="-1.5 -1.5 3 3" className="absolute inset-0">
        <defs>
          <radialGradient id="orbCore">
            <stop offset="0" stopColor="#fdf4ff" stopOpacity="0.95" />
            <stop offset="0.35" stopColor="#e879f9" stopOpacity="0.55" />
            <stop offset="0.7" stopColor="#7c3aed" stopOpacity="0.18" />
            <stop offset="1" stopColor="#7c3aed" stopOpacity="0" />
          </radialGradient>
          <filter id="orbGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="0.014" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <circle r="0.42" fill="url(#orbCore)">
          <animate attributeName="r" values="0.4;0.46;0.4" dur="6s" repeatCount="indefinite" />
        </circle>
        {ORB_RINGS.map(({ d, dot }, i) => (
          <g key={i}>
            {/* SMIL rotates about the user-space origin, the orb's centre, in every browser */}
            <animateTransform attributeName="transform" type="rotate" from={i % 2 ? "360 0 0" : "0 0 0"} to={i % 2 ? "0 0 0" : "360 0 0"}
                              dur={`${46 + i * 7}s`} repeatCount="indefinite" />
            <path d={d} fill="none" stroke={`hsl(${258 + i * 4} 92% ${68 + i * 1.2}%)`}
                  strokeOpacity={0.24 + 0.6 * Math.sin((Math.PI * (i + 1)) / 17)} strokeWidth={i % 4 === 0 ? 0.009 : 0.005}
                  filter={i % 4 === 0 ? "url(#orbGlow)" : undefined} />
            {i % 5 === 2 && <circle cx={dot[0]} cy={dot[1]} r="0.018" fill="#fdf4ff" filter="url(#orbGlow)" />}
          </g>
        ))}
      </svg>
    </div>
  );
}

function UploadStage({ files, setFiles, onRun, online, h1a, reference, onStudy }) {
  const drop = useRef(null);
  return (
    <>
      <section className="relative pt-4">
        <div className="relative h-[560px]">
          <div className="fade-up absolute left-1/2 top-0 -translate-x-1/2">
            <NeuralOrb size={540} />
          </div>

          {h1a && (
            <Glass strong className="fade-up float absolute left-0 top-16 hidden w-[290px] p-5 lg:block" style={{ animationDelay: "250ms" }}>
              <div className="flex items-center justify-between text-[13px] font-medium text-white">Pre-registered result<Chip tone="violet">H1a</Chip></div>
              <div className="mt-3 flex items-baseline gap-2">
                <span className="text-[38px] font-light tracking-tight text-white">{h1a.auc.toFixed(3)}</span>
                <span className="text-[12px] text-emerald-300">AUC · positive</span>
              </div>
              <div className="mt-3 rounded-xl bg-white/[0.05] p-3 text-[12px] leading-relaxed text-violet-100/60">
                ADHD versus control in an independent cohort of {h1a.n} children, with a model fixed before the data was opened.
              </div>
            </Glass>
          )}

          <Glass strong className="fade-up float absolute right-0 top-40 hidden w-[280px] p-5 lg:block" style={{ animationDelay: "400ms", animationDuration: "0.7s, 8s" }}>
            <div className="text-[13px] font-medium text-white">Analysis pipeline</div>
            <div className="mt-3 space-y-2.5">
              {[["Channels", "19 · 10-20 system"], ["Epochs", "1.5 s windows"], ["Images", "Wavelet scalograms"], ["Model", "YOLOv8n-cls"]].map(([k, v]) => (
                <div key={k} className="flex justify-between text-[12px]"><span className="text-violet-100/50">{k}</span><span className="text-white">{v}</span></div>
              ))}
            </div>
          </Glass>

          {reference.length > 0 && (
            <Glass strong className="fade-up float absolute bottom-24 left-10 hidden w-[230px] p-4 lg:block" style={{ animationDelay: "550ms", animationDuration: "0.7s, 9s" }}>
              <div className="text-[12px] text-violet-100/60">Reference cohort</div>
              <div className="mt-1 text-[22px] font-light text-white">{reference.length} children</div>
              <div className="text-[11px] text-violet-100/45">never seen by the classifier</div>
            </Glass>
          )}

          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-56 bg-linear-to-t from-[#07050f] via-[#07050f]/80 to-transparent" />
          <div className="absolute inset-x-0 bottom-0 text-center">
            <div className="fade-up" style={{ animationDelay: "150ms" }}>
              <Title className="mx-auto max-w-3xl text-[58px]">
                Reading a child's brain signal with <Em>intelligent image models</Em>
              </Title>
            </div>
          </div>
        </div>
        <div className="fade-up mx-auto mt-6 max-w-xl text-center text-[16px] leading-relaxed text-violet-100/60" style={{ animationDelay: "250ms" }}>
          Upload a resting-state EEG. It is cleaned, turned into time–frequency images and scored by a
          YOLOv8 classifier, live, on this machine.
        </div>
        <div className="fade-up mt-8 flex justify-center gap-3" style={{ animationDelay: "350ms" }}>
          <PrimaryButton onClick={() => drop.current?.scrollIntoView({ behavior: "smooth", block: "center" })}>
            Upload EEG <ArrowDown size={15} />
          </PrimaryButton>
          <GhostButton onClick={onStudy}>View the study</GhostButton>
        </div>
      </section>

      <section ref={drop} className="mt-24 scroll-mt-10">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="text-[12px] font-medium uppercase tracking-[0.22em] text-violet-300/80">Step 1 · Input</div>
            <div className="mt-2 text-[28px] tracking-tight text-white">Add the <Em>recording</Em></div>
          </div>
          <div className="text-[13px] text-violet-100/50">Files stay on this machine and are never stored.</div>
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          <Dropzone n="01" label="Resting-state recording" tag="EOEC · required" hint="Eyes closed, then eyes open, in one .edf file"
                    file={files.eoec} onFile={(f) => setFiles((s) => ({ ...s, eoec: f }))}
                    onClear={() => setFiles((s) => ({ ...s, eoec: null }))} />
          <Dropzone n="02" label="Task recording" tag="VCPT · optional" hint="Adds task biomarkers; the prediction uses rest only"
                    file={files.vcpt} onFile={(f) => setFiles((s) => ({ ...s, vcpt: f }))}
                    onClear={() => setFiles((s) => ({ ...s, vcpt: null }))} />
        </div>
        <Glass className="mt-5 flex flex-wrap items-center justify-between gap-4 p-4 pl-6">
          <div className="flex items-center gap-3 text-[13px] text-violet-100/60">
            <Info size={15} className="text-violet-300" />
            {online === false ? "The analysis server is offline. Start it with uvicorn first."
              : files.eoec ? "Ready. The analysis takes about 20 seconds."
                : "Add the resting-state recording to begin."}
          </div>
          <PrimaryButton onClick={onRun} disabled={!files.eoec || online === false}>
            Run analysis <ArrowRight size={15} className="transition group-hover:translate-x-0.5" />
          </PrimaryButton>
        </Glass>
      </section>
    </>
  );
}

function Dropzone({ n, label, tag, hint, file, onFile, onClear }) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState(null);
  const input = useRef(null);

  function take(f) {
    if (!f) return;
    if (!/\.edf$/i.test(f.name)) { setError("Choose an .edf file."); return; }
    setError(null);
    onFile(f);
  }

  if (file) {
    return (
      <div className="glass-strong relative flex min-h-[230px] flex-col rounded-[22px] p-7 ring-1 ring-violet-400/30">
        <div className="flex items-start justify-between">
          <span className="text-[12px] font-medium tracking-wider text-violet-300">{n}</span>
          <button type="button" onClick={onClear} aria-label="Remove file"
                  className="rounded-lg p-1.5 text-violet-100/50 transition hover:bg-white/10 hover:text-white"><X size={16} /></button>
        </div>
        <div className="mt-auto">
          <div className="relative inline-grid h-12 w-12 place-items-center rounded-2xl bg-violet-500/20 text-violet-200 ring-1 ring-violet-300/30">
            <FileText size={21} />
            <span className="absolute -bottom-1.5 -right-1.5 grid h-5 w-5 place-items-center rounded-full bg-emerald-400 text-[#07050f]"><Check size={11} strokeWidth={3.5} /></span>
          </div>
          <div className="mt-5 text-[17px] font-medium text-white">{label}</div>
          <div className="mt-1 truncate text-[13px] text-violet-100/55" title={file.name}>{file.name}</div>
          <div className="mt-4 flex gap-2"><Chip tone="ok">Loaded</Chip><Chip>{(file.size / 1e6).toFixed(1)} MB</Chip></div>
        </div>
      </div>
    );
  }

  return (
    <div role="button" tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}
      className={`group relative flex min-h-[230px] cursor-pointer flex-col rounded-[22px] border border-dashed p-7 transition-all duration-300
        ${over ? "scale-[1.01] border-violet-300 bg-violet-500/15" : "border-white/15 bg-white/[0.02] hover:-translate-y-1 hover:border-violet-300/60 hover:bg-violet-500/[0.07] hover:shadow-[0_20px_60px_-20px_rgba(139,92,246,0.5)]"}`}>
      <input ref={input} type="file" accept=".edf,.EDF" className="hidden" onChange={(e) => take(e.target.files[0])} />
      <div className="flex items-start justify-between">
        <span className="text-[12px] font-medium tracking-wider text-violet-100/40">{n}</span>
        <span className="text-[11px] font-medium uppercase tracking-wider text-violet-100/40">{tag}</span>
      </div>
      <div className="mt-auto">
        <span className={`grid h-12 w-12 place-items-center rounded-2xl transition-all duration-300 ${over ? "bg-white text-[#120a24]" : "bg-white/[0.06] text-violet-200 ring-1 ring-white/10 group-hover:bg-white group-hover:text-[#120a24]"}`}>
          <Upload size={20} />
        </span>
        <div className="mt-5 text-[17px] font-medium text-white">{label}</div>
        <div className="mt-1 text-[13px] text-violet-100/50">{hint}</div>
        <div className="mt-4 text-[13px] text-violet-100/40">
          Drop an <b className="font-medium text-violet-100/80">.edf</b> file or <span className="text-violet-300 underline-offset-4 group-hover:underline">browse</span>
        </div>
        {error && <div className="mt-2 text-xs text-rose-300">{error}</div>}
      </div>
    </div>
  );
}

/* ========================================================= 2. generate */

function GenerateStage({ run, elapsed, subjectId, onNext, onBack }) {
  const result = run.result;
  const running = run.status === "running";
  return (
    <>
      <PageHead eyebrow="Step 2 · Visualisation" title={<>From brain signal <Em>to images</Em></>}
        right={run.status === "done" && <PrimaryButton onClick={onNext}>See the result <ArrowRight size={15} className="transition group-hover:translate-x-0.5" /></PrimaryButton>}>
        The cleaned EEG is cut into 1.5 s epochs. Each becomes a time–frequency scalogram, the image the
        classifier reads; scalp maps and Grad-CAM show what is in it.
      </PageHead>

      {run.status === "error" && (
        <Glass className="fade-up mb-6 flex items-start gap-4 p-6 ring-1 ring-rose-400/30">
          <AlertTriangle className="mt-0.5 text-rose-300" size={20} />
          <div className="flex-1">
            <div className="font-medium text-white">The analysis did not complete</div>
            <div className="mt-1 text-sm text-rose-200/80">{run.error}</div>
          </div>
          <GhostButton onClick={onBack}><RotateCcw size={14} /> Start again</GhostButton>
        </Glass>
      )}

      <PipelineCard run={run} elapsed={elapsed} subjectId={subjectId} />

      {running && (
        <div className="mt-6 space-y-6">
          <div className="skeleton h-[420px] rounded-[22px]" />
          <div className="grid grid-cols-4 gap-5">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton aspect-[3/4] rounded-[22px]" />)}</div>
        </div>
      )}

      {result && (
        <div className="mt-6 space-y-6">
          <SignalSection result={result} />
          <ScalogramSection result={result} />
          <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
            <TopomapSection result={result} />
            <GradCamSection result={result} />
          </div>
          <Glass strong className="fade-up flex flex-wrap items-center justify-between gap-4 p-6 pl-8" style={{ animationDelay: "320ms" }}>
            <div>
              <div className="font-medium text-white">{result.n_epochs_analyzed} epochs scored</div>
              <div className="text-sm text-violet-100/50">Their mean is the child's probability.</div>
            </div>
            <PrimaryButton onClick={onNext}>See the result <ArrowRight size={15} className="transition group-hover:translate-x-0.5" /></PrimaryButton>
          </Glass>
        </div>
      )}
    </>
  );
}

function PipelineCard({ run, elapsed, subjectId }) {
  const running = run.status === "running";
  const done = run.status === "done";
  const activeIdx = PIPELINE.reduce((acc, p, i) => (elapsed >= p.at ? i : acc), 0);
  return (
    <Glass strong className="fade-up overflow-hidden">
      <div className="grid items-stretch md:grid-cols-[250px_1fr]">
        <div className="flex flex-col items-center justify-center border-white/10 bg-violet-500/[0.07] p-8 md:border-r">
          <div className="relative grid h-28 w-28 place-items-center">
            <svg viewBox="0 0 120 120" className={`absolute inset-0 ${running ? "spin-slow" : ""}`}>
              <defs><linearGradient id="ring" x1="0" x2="1"><stop offset="0" stopColor="#c4b5fd" /><stop offset="1" stopColor="#7c3aed" /></linearGradient></defs>
              <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="6" />
              <circle cx="60" cy="60" r="52" fill="none" stroke="url(#ring)" strokeWidth="6" strokeLinecap="round"
                      pathLength="100" strokeDasharray={done ? "100 100" : "28 100"} transform="rotate(-90 60 60)" />
            </svg>
            {done ? <Check size={34} strokeWidth={2.2} className="text-emerald-300" />
              : <div className="text-3xl font-light tabular-nums text-white">{elapsed.toFixed(0)}<span className="text-base text-violet-200/50">s</span></div>}
          </div>
          <div className="mt-5 text-center">
            <div className="text-sm font-medium text-white">{done ? "Analysis complete" : run.status === "error" ? "Stopped" : "Analysing"}</div>
            <div className="mt-0.5 text-xs text-violet-100/50">{subjectId || "Recording"}</div>
          </div>
        </div>
        <div className="p-8">
          <div className="mb-7 flex items-center justify-between">
            <div className="text-[15px] font-medium text-white">Processing pipeline</div>
            <Chip tone={done ? "ok" : run.status === "error" ? "bad" : "violet"}>
              {done ? `${run.result.n_epochs_analyzed} epochs classified` : run.status === "error" ? "Error" : "Typical timing"}
            </Chip>
          </div>
          <ol className="grid grid-cols-3 gap-x-4 gap-y-6 xl:grid-cols-6">
            {PIPELINE.map((p, i) => {
              const state = done ? "done" : !running ? "idle" : i < activeIdx ? "done" : i === activeIdx ? "active" : "idle";
              return (
                <li key={p.label} className="relative">
                  {i < PIPELINE.length - 1 && (
                    <span className="absolute left-11 right-1 top-[17px] hidden h-px bg-white/10 xl:block">
                      <span className={`block h-full bg-violet-300 transition-all duration-700 ${state === "done" ? "w-full" : "w-0"}`} />
                    </span>
                  )}
                  <span className={`relative grid h-9 w-9 place-items-center rounded-xl text-xs font-medium transition-all duration-500
                    ${state === "done" ? "bg-white text-[#120a24]" : state === "active" ? "bg-violet-500/25 text-white ring-1 ring-violet-300 shadow-[0_0_24px_4px_rgba(167,139,250,0.45)]" : "bg-white/[0.05] text-violet-100/40 ring-1 ring-white/10"}`}>
                    {state === "done" ? <Check size={15} strokeWidth={2.5} /> : i + 1}
                  </span>
                  <div className={`mt-3 text-sm font-medium ${state === "idle" ? "text-violet-100/40" : "text-white"}`}>{p.label}</div>
                  <div className="text-xs text-violet-100/40">{p.detail}</div>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </Glass>
  );
}

function SectionLabel({ n, title, hint }) {
  return (
    <div className="mb-5 flex flex-wrap items-baseline gap-x-3">
      <span className="font-serif text-[22px] italic text-violet-300">{n}</span>
      <span className="text-[18px] font-medium tracking-tight text-white">{title}</span>
      {hint && <span className="text-[13px] text-violet-100/45">{hint}</span>}
    </div>
  );
}

function SignalSection({ result }) {
  const tasks = Object.keys(result.signal || {});
  const [task, setTask] = useState(tasks[0] || "EC");
  const s = result.signal?.[task];
  if (!s) return null;
  return (
    <Glass className="fade-up p-7">
      <div className="mb-2 flex flex-wrap items-start justify-between gap-4">
        <SectionLabel n="i." title="Cleaned EEG signal" hint={`${s.channels.length} channels · ${s.n_epochs} epochs of 1.5 s · after filtering and ICA`} />
        <Toggle value={task} onChange={setTask} options={tasks.map((t) => [t, CONDITION[t]])} />
      </div>
      <EEGViewer key={task} signal={s} />
    </Glass>
  );
}

function EEGViewer({ signal }) {
  const W = 1060, rowH = 24, gutter = 44, top = 4;
  const n = signal.uv[0].length;
  const H = top + signal.channels.length * rowH + 24;
  const scale = useMemo(() => {
    const all = signal.uv.flat().map(Math.abs).sort((a, b) => a - b);
    return (rowH * 0.5) / (all[Math.floor(all.length * 0.95)] || 1);
  }, [signal]);
  const x = (i) => gutter + (i / (n - 1)) * (W - gutter - 6);
  const perEpoch = n / signal.n_epochs;
  const clamp = rowH * 0.65;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="EEG signal">
      <defs>
        <filter id="glow" x="-5%" y="-50%" width="110%" height="200%"><feGaussianBlur stdDeviation="1.4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
      </defs>
      {Array.from({ length: signal.n_epochs + 1 }, (_, k) => (
        <line key={`e${k}`} x1={x(Math.min(n - 1, k * perEpoch))} x2={x(Math.min(n - 1, k * perEpoch))}
              y1={top} y2={top + signal.channels.length * rowH} stroke="rgba(196,181,253,0.12)" strokeDasharray="3 5" />
      ))}
      {signal.channels.map((ch, r) => {
        const mid = top + r * rowH + rowH / 2;
        let d = "";
        signal.uv[r].forEach((v, i) => {
          d += `${i ? "L" : "M"}${x(i).toFixed(1)} ${(mid - Math.max(-clamp, Math.min(clamp, v * scale))).toFixed(1)}`;
        });
        const t = r / (signal.channels.length - 1);
        return (
          <g key={ch}>
            <text x={gutter - 10} y={mid + 3.5} fontSize="10" fontWeight="500" fill="rgba(221,214,254,0.55)" textAnchor="end">{ch}</text>
            <path d={d} pathLength="1" className="draw" style={{ animationDelay: `${r * 45}ms` }} filter="url(#glow)"
                  fill="none" stroke={`hsl(${258 + t * 52} 92% ${72 + t * 8}%)`} strokeWidth="1.05" strokeLinejoin="round" />
          </g>
        );
      })}
      {Array.from({ length: signal.n_epochs + 1 }, (_, k) => (
        <text key={`t${k}`} x={x(Math.min(n - 1, k * perEpoch))} y={H - 6} fontSize="10" fill="rgba(221,214,254,0.4)" textAnchor="middle">{(k * 1.5).toFixed(1)} s</text>
      ))}
    </svg>
  );
}

function MiniTrace({ uv, delay = 0 }) {
  const W = 300, rowH = 15, n = uv[0].length;
  const all = uv.flat().map(Math.abs).sort((a, b) => a - b);
  const scale = (rowH * 0.5) / (all[Math.floor(all.length * 0.95)] || 1);
  return (
    <svg viewBox={`0 0 ${W} ${uv.length * rowH + 4}`} className="w-full" aria-hidden="true">
      {uv.map((row, r) => {
        const mid = 2 + r * rowH + rowH / 2;
        const d = row.map((v, i) => `${i ? "L" : "M"}${((i / (n - 1)) * W).toFixed(1)} ${(mid - Math.max(-rowH * 0.6, Math.min(rowH * 0.6, v * scale))).toFixed(1)}`).join("");
        return <path key={r} d={d} pathLength="1" className="draw" style={{ animationDelay: `${delay + r * 60}ms` }}
                     fill="none" stroke={`hsl(${262 + r * 10} 90% 76%)`} strokeWidth="1.1" />;
      })}
    </svg>
  );
}

function ScalogramSection({ result }) {
  const [task, setTask] = useState("EC");
  const shown = result.images.scalograms.filter((s) => s.task === task);
  const channels = result.images.scalogram_channels || [];
  return (
    <Glass className="fade-up p-7" style={{ animationDelay: "120ms" }}>
      <div className="mb-2 flex flex-wrap items-start justify-between gap-4">
        <SectionLabel n="ii." title="Signal → time–frequency image" hint="Continuous wavelet transform of each epoch" />
        <Toggle value={task} onChange={setTask} options={[["EC", "Eyes closed"], ["EO", "Eyes open"]]} />
      </div>
      <div key={task} className="grid grid-cols-2 gap-5 xl:grid-cols-4">
        {shown.map((s, i) => (
          <figure key={s.epoch} className="fade-up group rounded-2xl bg-black/30 p-3 ring-1 ring-white/[0.07] transition hover:-translate-y-1 hover:ring-violet-300/40"
                  style={{ animationDelay: `${i * 90}ms` }}>
            <div className="rounded-xl bg-white/[0.03] px-2 py-2">{s.signal_uv && <MiniTrace uv={s.signal_uv} delay={i * 90} />}</div>
            <div className="flex items-center justify-center gap-1.5 py-2 text-[10px] font-medium uppercase tracking-[0.2em] text-violet-100/35"><ArrowDown size={11} /> wavelet</div>
            <img src={png(s.png_base64)} alt={`Scalogram, ${CONDITION[task]}, epoch ${s.epoch}`} className="aspect-square w-full rounded-xl object-cover" />
            <figcaption className="mt-3 flex items-center justify-between px-1">
              <span className="text-xs text-violet-100/50">Epoch {s.epoch}</span>
              <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium tabular-nums ${s.p_adhd >= 0.5 ? "bg-violet-400/20 text-violet-100" : "bg-white/[0.06] text-violet-100/60"}`}>p {s.p_adhd.toFixed(2)}</span>
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="mt-5 text-xs leading-relaxed text-violet-100/40">
        Channels {channels.join(", ")} stacked top to bottom; in each, 0.5–50 Hz from bottom to top and time left to right.
        Red: absolute power · green: power above the 1/f background · blue: power relative to each frequency.
        p is the classifier's ADHD probability for that single epoch.
      </p>
    </Glass>
  );
}

function TopomapSection({ result }) {
  return (
    <Glass className="fade-up p-7" style={{ animationDelay: "200ms" }}>
      <SectionLabel n="iii." title="Scalp maps" hint="Band power per epoch, as generated" />
      <div className="grid grid-cols-2 gap-5">
        {result.images.topomaps.map((t) => (
          <figure key={t.task}>
            <img src={png(t.png_base64)} alt={`Topographic map, ${CONDITION[t.task]}`} className="aspect-square w-full rounded-2xl bg-white object-contain p-2" />
            <figcaption className="mt-3 flex items-center justify-between text-sm">
              <span className="font-medium text-white">{CONDITION[t.task]}</span><span className="text-xs text-violet-100/40">epoch {t.epoch}</span>
            </figcaption>
          </figure>
        ))}
      </div>
      <div className="mt-5 flex flex-wrap gap-1.5">{BANDS.map(([b, l]) => <Chip key={b}>{l}</Chip>)}</div>
    </Glass>
  );
}

function GradCamSection({ result }) {
  return (
    <Glass className="fade-up p-7" style={{ animationDelay: "260ms" }}>
      <SectionLabel n="iv." title="Where the model looked" hint="Grad-CAM" />
      <img src={png(result.gradcam_overlay_png_base64)} alt="Grad-CAM overlay" className="aspect-square w-full rounded-2xl object-cover" />
      <p className="mt-4 text-xs leading-relaxed text-violet-100/50">
        Warm regions of this scalogram drove the classifier's score most. It describes the model, not the child:
        a way to inspect what the network uses, not proof that the pattern is physiological.
      </p>
    </Glass>
  );
}

/* ========================================================== 3. results */

function ResultsStage({ result, reference, subjectId, onStudy }) {
  const [band, setBand] = useState("alpha");
  const [cond, setCond] = useState("ec");
  const bio = result.classical_biomarkers;
  const values = useMemo(() => Object.fromEntries(Object.keys(POS).map((ch) => [ch, bio[`relpow_${ch}_${band}_${cond}`]])), [bio, band, cond]);
  const isAdhd = result.predicted_class === "ADHD";
  const others = reference.filter((r) => r.subject_id !== subjectId);
  const above = (g) => others.filter((r) => r.group === g && r.p_adhd >= result.p_adhd).length;
  const total = (g) => others.filter((r) => r.group === g).length;
  const qc = result.quality_control;

  return (
    <>
      <section className="pt-10 text-center">
        <div className="fade-up text-[12px] font-medium uppercase tracking-[0.22em] text-violet-300/80">Step 3 · Results · {subjectId}</div>
        <div className="fade-up" style={{ animationDelay: "100ms" }}>
          <Title className="mx-auto mt-4 max-w-3xl text-[54px]">
            This recording shows {isAdhd ? <>an <Em>ADHD-like pattern</Em></> : <>a <Em>control-like pattern</Em></>}
          </Title>
        </div>
      </section>

      <section className="relative mt-8 lg:h-[620px]">
        <div className="fade-up lg:absolute lg:left-1/2 lg:top-0 lg:-translate-x-1/2" style={{ animationDelay: "150ms" }}>
          <HeadMap values={values} size={520} rays legend />
          <div className="mt-8 flex flex-wrap justify-center gap-2">
            <Toggle value={band} onChange={setBand} options={BANDS} />
            <Toggle value={cond} onChange={setCond} options={[["ec", "Closed"], ["eo", "Open"]]} />
          </div>
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:mt-0">
          <ProbabilityCard result={result} className="lg:absolute lg:left-0 lg:top-4 lg:w-[290px]" />
          <Glass strong className="fade-up float p-5 lg:absolute lg:right-0 lg:top-10 lg:w-[270px]" style={{ animationDelay: "350ms", animationDuration: "0.7s, 8s" }}>
            <div className="text-[13px] font-medium text-white">Among unseen children</div>
            <div className="mt-4 space-y-3">
              {[["ADHD", COLORS.adhd], ["Control", "#8b93ad"]].map(([g, c]) => (
                <div key={g} className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-[12px] text-violet-100/60"><i className="h-2 w-2 rounded-full" style={{ background: c }} />{g} at or above</span>
                  <span className="text-[18px] font-light tabular-nums text-white">{above(g)}<span className="text-[12px] text-violet-100/40"> / {total(g)}</span></span>
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-xl bg-white/[0.05] p-3 text-[11px] leading-relaxed text-violet-100/50">
              Children the classifier never trained on, scored by this same pipeline.
            </div>
          </Glass>
          <Glass strong className="fade-up float p-5 lg:absolute lg:bottom-24 lg:left-6 lg:w-[250px]" style={{ animationDelay: "450ms", animationDuration: "0.7s, 9s" }}>
            <div className="flex items-center justify-between text-[13px] font-medium text-white">
              Recording quality <Chip tone={qc?.flagged ? "warn" : "ok"}>{qc?.flagged ? "Review" : "Passed"}</Chip>
            </div>
            <div className="mt-4 space-y-2.5">
              {Object.entries(qc?.conditions || {}).map(([t, c]) => (
                <div key={t}>
                  <div className="mb-1 flex justify-between text-[11px]"><span className="text-violet-100/55">{CONDITION[t]}</span><span className="tabular-nums text-white">{pct(1 - c.rejection_rate)} kept</span></div>
                  <div className="h-1 rounded-full bg-white/10"><div className={`grow-x h-full rounded-full ${c.high_rejection ? "bg-amber-300" : "bg-emerald-300"}`} style={{ width: pct(1 - c.rejection_rate) }} /></div>
                </div>
              ))}
            </div>
          </Glass>
          <Glass strong className="fade-up float p-5 lg:absolute lg:bottom-28 lg:right-6 lg:w-[250px]" style={{ animationDelay: "550ms", animationDuration: "0.7s, 7.5s" }}>
            <div className="text-[13px] font-medium text-white">This child's scalp map</div>
            <div className="mt-2 text-[12px] leading-relaxed text-violet-100/55">
              Relative {band} power, {cond === "ec" ? "eyes closed" : "eyes open"}, interpolated between the 19 electrodes.
              Change the band below the map.
            </div>
          </Glass>
        </div>
      </section>

      <Glass className="fade-up mt-16 p-7">
        <CardHead icon={BarChart3} title="Reference distribution"
          hint={`This recording among ${reference.length} children the model never trained on, scored by the same pipeline.`} />
        <ReferenceStrip reference={reference} value={result.p_adhd} subjectId={subjectId} />
      </Glass>

      <div className="mt-6 grid gap-5 sm:grid-cols-2 xl:grid-cols-5">
        {BIOMARKERS.map((b, i) => <BiomarkerTile key={b.key} b={b} values={bio} delay={100 + i * 60} />)}
        <Glass className="fade-up flex flex-col p-6" style={{ animationDelay: "340ms" }}>
          <div className="min-h-[2.75rem]"><div className="text-sm font-medium text-white">Alpha reactivity</div><div className="text-[11px] text-violet-100/45">Closed → open</div></div>
          <div className="mt-3 text-[34px] font-light leading-none tracking-tight tabular-nums text-white">{fmt(bio.alpha_reactivity, 2)}</div>
          <div className="mt-1 text-[11px] text-violet-100/45">ratio</div>
          <p className="mt-auto pt-5 text-[11px] leading-relaxed text-violet-100/45">How strongly occipital alpha falls when the eyes open.</p>
        </Glass>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Glass className="fade-up p-7" style={{ animationDelay: "200ms" }}>
          <CardHead icon={Activity} title="Epoch-by-epoch scores" hint="The classifier's score for every classified epoch; the child's score is their mean." />
          <EpochHistogram epochs={result.epoch_probabilities} mean={result.p_adhd} />
        </Glass>
        {qc && <QualityCard qc={qc} />}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="fade-up flex gap-4 rounded-[22px] border border-amber-300/20 bg-amber-300/[0.06] p-7">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-300/15 text-amber-200"><ShieldAlert size={19} /></span>
          <div>
            <div className="font-medium text-amber-50">This is a decision-support research tool, not a diagnostic device.</div>
            <p className="mt-1.5 text-sm leading-relaxed text-amber-100/60">{result.disclaimer}</p>
          </div>
        </div>
        <button type="button" onClick={onStudy}
          className="fade-up glass-strong group flex items-center justify-between gap-4 rounded-[22px] p-7 text-left transition hover:-translate-y-1">
          <span>
            <span className="block text-[12px] font-medium uppercase tracking-[0.2em] text-violet-300/80">Step 4 · Validation</span>
            <span className="mt-2 block text-[18px] text-white">Does it hold up? <Em>The study.</Em></span>
          </span>
          <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-white text-[#120a24] transition group-hover:translate-x-1"><ArrowRight size={18} /></span>
        </button>
      </div>
    </>
  );
}

function ProbabilityCard({ result, className = "" }) {
  const p = result.p_adhd;
  const v = useCountUp(p);
  const R = 46, C = 2 * Math.PI * R;
  return (
    <Glass strong className={`fade-up float p-5 ${className}`} style={{ animationDelay: "250ms" }}>
      <div className="flex items-center justify-between text-[13px] font-medium text-white">
        Probability of ADHD <Chip tone="violet">{result.predicted_class}</Chip>
      </div>
      <div className="mt-4 flex items-center gap-5">
        <div className="relative h-[108px] w-[108px] shrink-0">
          <svg viewBox="0 0 108 108" className="absolute inset-0">
            <defs><linearGradient id="pg" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stopColor="#c4b5fd" /><stop offset="1" stopColor="#e879f9" /></linearGradient></defs>
            <circle cx="54" cy="54" r={R} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="9" />
            <circle cx="54" cy="54" r={R} fill="none" stroke="url(#pg)" strokeWidth="9" strokeLinecap="round"
                    strokeDasharray={`${C * v} ${C}`} transform="rotate(-90 54 54)" />
            {/* the 0.5 decision threshold, half way round the ring */}
            <line x1="54" y1="96" x2="54" y2="106" stroke="#fff" strokeOpacity="0.4" strokeWidth="1.5" />
          </svg>
          <div className="absolute inset-0 grid place-items-center text-[30px] font-light tabular-nums text-white">{v.toFixed(2)}</div>
        </div>
        <div className="space-y-2 text-[12px]">
          <div><div className="text-violet-100/45">Epochs</div><div className="text-white">{result.n_epochs_analyzed}</div></div>
          <div><div className="text-violet-100/45">Confidence</div><div className="text-white">{pct(result.confidence)}</div></div>
          <div><div className="text-violet-100/45">Threshold</div><div className="text-white">0.50</div></div>
        </div>
      </div>
    </Glass>
  );
}

function ReferenceStrip({ reference, value, subjectId }) {
  const W = 1060, L = 90, R = 20, ROWS = [{ group: "ADHD", y: 70 }, { group: "Control", y: 156 }];
  const x = (v) => L + v * (W - L - R);
  const lanes = [0, -11, 11, -22, 22];
  const placed = useMemo(() => ROWS.flatMap((row) => {
    const taken = lanes.map(() => []);
    return reference.filter((r) => r.group === row.group).sort((a, b) => a.p_adhd - b.p_adhd).map((r) => {
      let lane = taken.findIndex((xs) => xs.every((u) => Math.abs(u - r.p_adhd) > 0.014));
      if (lane < 0) lane = 0;
      taken[lane].push(r.p_adhd);
      return { ...r, cx: x(r.p_adhd), cy: row.y + lanes[lane] };
    });
  }), [reference]); // eslint-disable-line react-hooks/exhaustive-deps
  const labelX = Math.min(Math.max(x(value), L + 72), W - R - 72);
  return (
    <svg viewBox={`0 0 ${W} 222`} className="w-full" role="img" aria-label="Reference distribution">
      <defs><filter id="dotglow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
      {ROWS.map((row) => (
        <g key={row.group}>
          <rect x={L - 10} y={row.y - 30} width={W - L - R + 20} height={60} rx={30} fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.06)" />
          <circle cx={10} cy={row.y} r={5} fill={row.group === "ADHD" ? COLORS.adhd : "#8b93ad"} />
          <text x={24} y={row.y + 4} fontSize="13" fontWeight="500" fill="rgba(237,233,254,0.85)">{row.group}</text>
        </g>
      ))}
      <line x1={x(0.5)} x2={x(0.5)} y1={34} y2={192} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 5" />
      {placed.map((d, i) => {
        const self = subjectId && d.subject_id === subjectId;
        return (
          <circle key={d.subject_id} cx={d.cx} cy={d.cy} r={self ? 8.5 : 6.5} className="pop" style={{ animationDelay: `${200 + i * 22}ms` }}
            fill={self ? COLORS.self : d.group === "ADHD" ? COLORS.adhd : COLORS.control} filter={self ? "url(#dotglow)" : undefined}
            stroke={self ? "#fff" : "rgba(7,5,15,0.8)"} strokeWidth={self ? 2.5 : 1.5}>
            <title>{`${d.subject_id} · ${d.group} · p ${d.p_adhd.toFixed(2)}`}</title>
          </circle>
        );
      })}
      <line x1={x(value)} x2={x(value)} y1={30} y2={194} stroke={COLORS.self} strokeWidth="2" />
      <rect x={labelX - 72} y={2} width={144} height={24} rx={12} fill="#ffffff" />
      <text x={labelX} y={18} fontSize="12" fontWeight="600" fill="#120a24" textAnchor="middle">This recording · {value.toFixed(2)}</text>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => <text key={t} x={x(t)} y={216} fontSize="11" fill="rgba(221,214,254,0.4)" textAnchor="middle">{t}</text>)}
    </svg>
  );
}

function BiomarkerTile({ b, values, delay }) {
  const ec = values[`${b.key}_ec`], eo = values[`${b.key}_eo`], task = values[`${b.key}_vcpt`];
  const max = Math.max(...[ec, eo, task].filter((v) => v !== null && v !== undefined).map(Math.abs), 1e-9);
  return (
    <Glass className="fade-up p-6" style={{ animationDelay: `${delay}ms` }}>
      <div className="min-h-[2.75rem]"><div className="text-sm font-medium leading-snug text-white">{b.label}</div><div className="text-[11px] text-violet-100/45">{b.hint}</div></div>
      <div className="mt-3 text-[34px] font-light leading-none tracking-tight tabular-nums text-white">{fmt(ec, b.d)}</div>
      <div className="mt-1 text-[11px] text-violet-100/45">eyes closed</div>
      <div className="mt-5 space-y-2">
        {[["EC", ec, COLORS.ec], ["EO", eo, COLORS.eo], ["Task", task, "#ddd6fe"]].map(([k, val, c]) => (
          <div key={k} className="flex items-center gap-2 text-[11px]">
            <span className="w-7 text-violet-100/40">{k}</span>
            <span className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
              {val !== null && val !== undefined && <span className="grow-x block h-full rounded-full" style={{ width: `${(Math.abs(val) / max) * 100}%`, background: c, animationDelay: `${delay + 200}ms` }} />}
            </span>
            <span className="w-9 text-right tabular-nums text-violet-100/75">{fmt(val, b.d)}</span>
          </div>
        ))}
      </div>
    </Glass>
  );
}

function EpochHistogram({ epochs, mean }) {
  const bins = 20;
  const counts = useMemo(() => {
    const c = Array.from({ length: bins }, () => ({ EC: 0, EO: 0 }));
    epochs.forEach((e) => { c[Math.min(bins - 1, Math.floor(e.p_adhd * bins))][e.task] += 1; });
    return c;
  }, [epochs]);
  const max = Math.max(...counts.map((c) => c.EC + c.EO), 1);
  return (
    <div>
      <div className="relative h-44">
        <div className="absolute inset-y-0 border-l border-dashed border-white/20" style={{ left: "50%" }} />
        <div className="absolute -top-1 bottom-0 z-10 w-0.5 rounded bg-fuchsia-200" style={{ left: pct(mean) }}>
          <span className="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-[#120a24]">mean {mean.toFixed(2)}</span>
        </div>
        <div className="flex h-full items-end gap-1">
          {counts.map((c, i) => (
            <div key={i} className="bar-grow flex flex-1 flex-col justify-end overflow-hidden rounded-t-md" style={{ height: `${((c.EC + c.EO) / max) * 100}%`, animationDelay: `${i * 30}ms` }}>
              <div style={{ flex: c.EO, background: COLORS.eo, opacity: 0.55 }} />
              <div style={{ flex: c.EC, background: COLORS.ec }} />
            </div>
          ))}
        </div>
      </div>
      <div className="mt-2 flex justify-between text-[11px] text-violet-100/40"><span>0 · Control-like</span><span>0.5</span><span>ADHD-like · 1</span></div>
      <div className="mt-4 flex gap-5 text-xs text-violet-100/55">
        <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-sm" style={{ background: COLORS.ec }} />Eyes closed</span>
        <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-sm" style={{ background: COLORS.eo, opacity: 0.55 }} />Eyes open</span>
        <span className="flex items-center gap-1.5"><i className="h-3 w-0.5 rounded bg-fuchsia-200" />Mean</span>
      </div>
    </div>
  );
}

function QualityCard({ qc }) {
  const tasks = Object.keys(qc.conditions || {});
  return (
    <Glass className="fade-up p-7" style={{ animationDelay: "260ms" }}>
      <CardHead icon={qc.flagged ? AlertTriangle : ShieldCheck} title="Recording quality" hint="Share of epochs kept after artifact rejection."
        right={<Chip tone={qc.flagged ? "warn" : "ok"}>{qc.flagged ? "Review advised" : "Passed"}</Chip>} />
      <div className="flex justify-around gap-4">
        {tasks.map((t) => {
          const c = qc.conditions[t]; const kept = 1 - c.rejection_rate; const R = 34, C = 2 * Math.PI * R;
          return (
            <div key={t} className="text-center">
              <div className="relative mx-auto h-[88px] w-[88px]">
                <svg viewBox="0 0 88 88" className="absolute inset-0">
                  <circle cx="44" cy="44" r={R} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="7" />
                  <circle cx="44" cy="44" r={R} fill="none" stroke={c.high_rejection ? "#fcd34d" : "#6ee7b7"} strokeWidth="7" strokeLinecap="round" strokeDasharray={`${C * kept} ${C}`} transform="rotate(-90 44 44)" />
                </svg>
                <div className="absolute inset-0 grid place-items-center text-sm font-medium tabular-nums text-white">{pct(kept)}</div>
              </div>
              <div className="mt-2 text-xs font-medium text-white">{CONDITION[t] || t}</div>
              <div className="text-[11px] text-violet-100/45">{c.n_epochs_surviving}/{c.n_epochs_recorded} kept</div>
            </div>
          );
        })}
      </div>
      {qc.notes?.length > 0 && (
        <details className="mt-6 border-t border-white/10 pt-4 text-xs text-violet-100/55">
          <summary className="cursor-pointer font-medium text-violet-100/75 hover:text-white">{qc.notes.length} note{qc.notes.length > 1 ? "s" : ""} from the pipeline</summary>
          <ul className="mt-3 space-y-2 leading-relaxed">{qc.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </details>
      )}
    </Glass>
  );
}

/* ======================================================= 4. validation */

const HYPOTHESES = {
  H1a: { title: "Within an independent cohort", sub: "Nasrabadi et al. · trained and tested by child, 5-fold" },
  H1b: { title: "Transfer between cohorts", sub: "Trained on Rohani et al., tested once on Nasrabadi et al." },
  H3: { title: "A new representation", sub: "Riemannian covariance features · Rohani et al." },
  H2: { title: "Held-out split", sub: "Post-hoc model · Rohani et al. · scored once" },
};

function ValidationStage({ study }) {
  const h1a = study.find((r) => r.hypothesis === "H1a");
  if (!h1a) return <Glass className="mt-12 p-7 text-sm text-violet-100/60">The study results could not be loaded from the analysis server.</Glass>;
  const order = ["H1a", "H1b", "H3", "H2"].map((h) => study.find((r) => r.hypothesis === h)).filter(Boolean);
  const confirmatory = study.filter((r) => ["H1a", "H1b", "H3"].includes(r.hypothesis));
  const positives = confirmatory.filter((r) => String(r.decision).toLowerCase() === "positive").length;
  return (
    <>
      <ValidationHero h1a={h1a} positives={positives} total={confirmatory.length} />
      <Glass className="fade-up mt-16 p-7" style={{ animationDelay: "160ms" }}>
        <CardHead icon={BarChart3} title="All pre-registered results" hint={`${positives} of ${confirmatory.length} confirmatory tests positive · dashed line = chance (AUC 0.5)`} />
        <Forest rows={order} />
      </Glass>
      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <Glass className="fade-up relative overflow-hidden p-7" style={{ animationDelay: "240ms" }}>
          <span className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-violet-300 to-transparent" />
          <CardHead icon={Check} title="What the study shows" />
          <ul className="space-y-3 text-sm leading-relaxed text-violet-100/65">
            <li>EEG separates children with ADHD from controls within the Nasrabadi cohort, with a model fixed in advance.</li>
            <li>The pipeline detects real differences: every model separates rest from task in the same children.</li>
          </ul>
        </Glass>
        <Glass className="fade-up relative overflow-hidden p-7" style={{ animationDelay: "300ms" }}>
          <CardHead icon={Info} title="What it does not show" />
          <ul className="space-y-3 text-sm leading-relaxed text-violet-100/65">
            <li>A marker that holds across cohorts: the model trained on one did not transfer to the other.</li>
            <li>A diagnostic test: an AUC of {h1a.auc.toFixed(2)} is a group difference, far from individual diagnosis.</li>
          </ul>
        </Glass>
      </div>
    </>
  );
}

function ValidationHero({ h1a, positives, total }) {
  const v = useCountUp(h1a.auc, 1500);
  const lo = 0.3, hi = 0.9;
  const at = (u) => `${((u - lo) / (hi - lo)) * 100}%`;
  return (
    <section className="relative pt-10 text-center">
      <div className="fade-up text-[12px] font-medium uppercase tracking-[0.22em] text-violet-300/80">Step 4 · Pre-registered validation</div>
      <div className="fade-up" style={{ animationDelay: "100ms" }}>
        <Title className="mx-auto mt-4 max-w-3xl text-[52px]">
          A model fixed in advance separates ADHD in an <Em>independent cohort</Em>
        </Title>
      </div>

      <div className="relative mt-10 lg:h-[420px]">
        <div className="fade-up lg:absolute lg:left-1/2 lg:top-0 lg:-translate-x-1/2" style={{ animationDelay: "200ms" }}>
          <div className="relative mx-auto grid h-[380px] w-[380px] place-items-center">
            <svg viewBox="-1.5 -1.5 3 3" className="spin-very-slow absolute inset-0" aria-hidden="true">
              {Array.from({ length: 144 }, (_, i) => {
                const a = (i / 144) * 2 * Math.PI, r1 = 1.2, r2 = r1 + (i % 6 === 0 ? 0.13 : 0.06);
                return <line key={i} x1={r1 * Math.cos(a)} y1={r1 * Math.sin(a)} x2={r2 * Math.cos(a)} y2={r2 * Math.sin(a)} stroke="#c4b5fd" strokeOpacity={i % 6 === 0 ? 0.45 : 0.18} strokeWidth="0.008" />;
              })}
            </svg>
            <div className="absolute inset-[18%] rounded-full bg-[radial-gradient(closest-side,rgba(139,92,246,0.55),rgba(76,29,149,0.25)_60%,transparent)]" />
            <div className="relative">
              <div className="font-serif text-[120px] leading-none text-white">{v.toFixed(3)}</div>
              <div className="mt-2 text-[13px] uppercase tracking-[0.2em] text-violet-200/70">AUC · H1a</div>
            </div>
          </div>
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:mt-0">
          <Glass strong className="fade-up float p-5 text-left lg:absolute lg:left-0 lg:top-6 lg:w-[290px]" style={{ animationDelay: "300ms" }}>
            <div className="flex items-center justify-between text-[13px] font-medium text-white">95% confidence interval<Chip tone="ok">above chance</Chip></div>
            <div className="mt-3 text-[26px] font-light tabular-nums text-white">{h1a.ci_low.toFixed(3)} – {h1a.ci_high.toFixed(3)}</div>
            <div className="relative mt-4 h-6">
              <div className="absolute inset-x-0 top-1/2 h-px bg-white/15" />
              <div className="absolute inset-y-0 border-l border-dashed border-amber-300/80" style={{ left: at(0.5) }} />
              <div className="grow-x absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-linear-to-r from-violet-300 to-fuchsia-300" style={{ left: at(h1a.ci_low), width: `calc(${at(h1a.ci_high)} - ${at(h1a.ci_low)})`, animationDelay: "600ms" }} />
              <div className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[#120a24] bg-white" style={{ left: at(h1a.auc) }} />
            </div>
            <div className="mt-1 flex justify-between text-[10px] text-violet-100/40"><span>0.3</span><span>0.5 chance</span><span>0.9</span></div>
          </Glass>
          <Glass strong className="fade-up float p-5 text-left lg:absolute lg:right-0 lg:top-14 lg:w-[270px]" style={{ animationDelay: "400ms", animationDuration: "0.7s, 8s" }}>
            <div className="text-[13px] font-medium text-white">Significance</div>
            <div className="mt-3 flex items-baseline justify-between"><span className="text-[12px] text-violet-100/50">Permutation p</span><span className="text-[22px] font-light tabular-nums text-white">{h1a.perm_p.toFixed(4)}</span></div>
            <div className="mt-1 flex items-baseline justify-between"><span className="text-[12px] text-violet-100/50">Pass mark</span><span className="text-[15px] tabular-nums text-white">p &lt; 0.0167</span></div>
            <div className="mt-3 rounded-xl bg-white/[0.05] p-3 text-[11px] leading-relaxed text-violet-100/50">Bonferroni over three tests; the whole interval must be above chance.</div>
          </Glass>
          <Glass strong className="fade-up float p-5 text-left lg:absolute lg:bottom-0 lg:left-12 lg:w-[250px]" style={{ animationDelay: "500ms", animationDuration: "0.7s, 9s" }}>
            <div className="text-[13px] font-medium text-white">Independent data</div>
            <div className="mt-2 text-[22px] font-light text-white">{h1a.n} children</div>
            <div className="text-[11px] text-violet-100/50">{h1a.n_adhd} with ADHD · Nasrabadi et al. · opened only after the plan was committed</div>
          </Glass>
          <Glass strong className="fade-up float p-5 text-left lg:absolute lg:bottom-4 lg:right-10 lg:w-[250px]" style={{ animationDelay: "600ms", animationDuration: "0.7s, 7.5s" }}>
            <div className="text-[13px] font-medium text-white">Confirmatory family</div>
            <div className="mt-2 text-[22px] font-light text-white">{positives} of {total} positive</div>
            <div className="text-[11px] text-violet-100/50">Every result reported, whatever it was</div>
          </Glass>
        </div>
      </div>
    </section>
  );
}

function Forest({ rows }) {
  const lo = 0.2, hi = 0.9;
  const at = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const grid = "grid grid-cols-[minmax(0,15rem)_1fr_5rem_6.5rem] items-center gap-6";
  return (
    <div>
      {rows.map((r, i) => {
        const decision = String(r.decision).toLowerCase();
        const pos = decision === "positive";
        const est = decision !== "positive" && decision !== "negative";
        return (
          <div key={r.hypothesis} className={`${grid} rounded-2xl px-4 py-4 ${pos ? "bg-violet-500/[0.12] ring-1 ring-violet-300/25" : ""}`}>
            <div className="min-w-0">
              <div className="text-sm font-medium text-white"><span className="mr-2 font-serif italic text-violet-300">{r.hypothesis}</span>{HYPOTHESES[r.hypothesis]?.title}</div>
              <div className="mt-0.5 text-xs leading-relaxed text-violet-100/45">{HYPOTHESES[r.hypothesis]?.sub}</div>
            </div>
            <div className="relative h-7">
              <div className="absolute -bottom-3 -top-3 border-l border-dashed border-amber-300/70" style={{ left: at(0.5) }} />
              <div className={`grow-x absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full ${pos ? "bg-linear-to-r from-violet-300 to-fuchsia-300" : est ? "bg-white/15" : "bg-white/30"}`}
                   style={{ left: at(r.ci_low), width: `calc(${at(r.ci_high)} - ${at(r.ci_low)})`, animationDelay: `${300 + i * 120}ms` }} />
              <div className={`pop absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[#120a24] ${pos ? "bg-white shadow-[0_0_16px_3px_rgba(232,121,249,0.6)]" : est ? "bg-white/40" : "bg-white/70"}`}
                   style={{ left: at(r.auc), animationDelay: `${600 + i * 120}ms` }} />
            </div>
            <div className="text-sm tabular-nums"><div className="text-white">{r.auc.toFixed(3)}</div><div className="text-xs text-violet-100/45">{r.ci_low.toFixed(2)}–{r.ci_high.toFixed(2)}</div></div>
            <span className={`justify-self-start rounded-full px-3 py-1 text-[11px] font-medium uppercase tracking-wide ${pos ? "bg-white text-[#120a24]" : "bg-white/[0.06] text-violet-100/50"}`}>{pos ? "Positive" : est ? "Estimate" : "Negative"}</span>
          </div>
        );
      })}
      <div className={`${grid} px-4 pt-1`}>
        <div />
        <div className="relative h-4 text-[11px] text-violet-100/40">{[0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map((t) => <span key={t} className="absolute -translate-x-1/2" style={{ left: at(t) }}>{t.toFixed(1)}</span>)}</div>
        <div className="text-[11px] text-violet-100/40">AUC</div>
        <div />
      </div>
    </div>
  );
}
