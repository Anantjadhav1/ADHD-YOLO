import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDown, ArrowRight, BarChart3, Brain, Check, FileText, FlaskConical,
  Gauge, Info, Layers, RotateCcw, ShieldAlert, ShieldCheck, Sparkles, Upload, Waves, X,
} from "lucide-react";

// In development the page runs on Vite's port and talks to uvicorn on :8000; once built,
// the backend serves the page itself at /app/, so the API is the same origin.
const API = import.meta.env.DEV ? "http://localhost:8000" : "";

const STEPS = [
  { id: 1, label: "Upload", hint: "EEG recording", icon: Upload },
  { id: 2, label: "Generate", hint: "Signal to images", icon: Layers },
  { id: 3, label: "Results", hint: "Prediction and biomarkers", icon: Gauge },
  { id: 4, label: "Validation", hint: "Pre-registered study", icon: FlaskConical },
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

const CONDITION = { EC: "Eyes closed", EO: "Eyes open", VCPT: "Task" };
const COLORS = { adhd: "#3b6cf6", control: "#9aa8bd", ec: "#3b6cf6", eo: "#a5b4fc", ink: "#0b1530" };

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

  return (
    <div className="min-h-screen text-slate-900 lg:flex">
      <Sidebar step={step} reachable={reachable} done={run.status === "done"} onGo={setStep}
               subjectId={subjectId} files={files} online={online} run={run} />
      <div className="min-w-0 flex-1">
        <TopBar step={step} reachable={reachable} onGo={setStep} online={online}
                canReset={run.status !== "idle" && run.status !== "running"} onReset={reset} />
        <main className="mx-auto max-w-[1180px] px-6 pb-20 pt-8 lg:px-12">
          <div key={step}>
            {step === 1 && <UploadStage files={files} setFiles={setFiles} onRun={analyse} online={online} />}
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
        <footer className="mx-auto flex max-w-[1180px] flex-wrap justify-between gap-3 border-t border-slate-200 px-6 py-6 text-xs text-slate-400 lg:px-12">
          <span>ADHD-YOLO · EEG classification of ADHD in children · thesis research prototype</span>
          <span>Every value is computed live by the analysis server or read from the committed study files.</span>
        </footer>
      </div>
    </div>
  );
}

/* ================================================================ frame */

function Logo({ dark }) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative grid h-11 w-11 place-items-center rounded-2xl bg-linear-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-600/30">
        <Brain size={22} strokeWidth={1.8} />
      </div>
      <div>
        <div className={`text-[16px] font-semibold tracking-tight ${dark ? "text-white" : "text-slate-900"}`}>ADHD-YOLO</div>
        <div className={`text-xs ${dark ? "text-slate-400" : "text-slate-500"}`}>EEG research workstation</div>
      </div>
    </div>
  );
}

function Sidebar({ step, reachable, done, onGo, subjectId, files, online, run }) {
  return (
    <aside className="sticky top-0 hidden h-screen w-[292px] shrink-0 flex-col overflow-hidden bg-[#0b1530] text-slate-300 lg:flex">
      <div className="pointer-events-none absolute -left-24 -top-24 h-72 w-72 rounded-full bg-blue-600/25 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-32 -right-20 h-72 w-72 rounded-full bg-indigo-500/20 blur-3xl" />
      <div className="relative px-7 pt-8"><Logo dark /></div>

      <div className="relative mt-12 px-7">
        <div className="mb-5 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Workflow</div>
        <ol className="relative">
          <span className="absolute left-[19px] top-5 h-[calc(100%-40px)] w-px bg-white/10" />
          <span className="absolute left-[19px] top-5 w-px bg-linear-to-b from-blue-400 to-indigo-400 transition-all duration-700"
                style={{ height: `calc((100% - 40px) * ${(step - 1) / 3})` }} />
          {STEPS.map((s) => {
            const active = s.id === step;
            const complete = s.id < step || (s.id === 3 && done && step === 4);
            const Icon = s.icon;
            return (
              <li key={s.id}>
                <button type="button" disabled={!reachable[s.id]} onClick={() => onGo(s.id)}
                  className={`group relative flex w-full items-center gap-4 rounded-2xl py-3 pr-3 text-left transition-all disabled:cursor-not-allowed
                    ${active ? "" : "hover:bg-white/[0.04]"} disabled:opacity-40`}>
                  <span className={`relative z-10 grid h-10 w-10 shrink-0 place-items-center rounded-xl transition-all duration-500
                    ${active ? "bg-linear-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-600/40"
                      : complete ? "bg-blue-500/15 text-blue-300 ring-1 ring-blue-400/30" : "bg-[#121d3d] text-slate-500 ring-1 ring-white/10"}`}>
                    {complete && !active ? <Check size={17} strokeWidth={2.5} /> : <Icon size={17} />}
                  </span>
                  <span className="min-w-0">
                    <span className={`block text-sm font-semibold ${active ? "text-white" : "text-slate-300"}`}>{s.label}</span>
                    <span className="block truncate text-xs text-slate-500">{s.hint}</span>
                  </span>
                  {active && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-blue-400 shadow-[0_0_12px_3px_rgba(96,165,250,0.7)]" />}
                </button>
              </li>
            );
          })}
        </ol>
      </div>

      {files.eoec && (
        <div className="fade-up relative mx-5 mt-10 rounded-2xl bg-white/[0.05] p-5 ring-1 ring-white/10">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Current recording</div>
          <div className="mt-2 text-lg font-semibold tracking-tight text-white">{subjectId}</div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            <Chip dark>Rest · EOEC</Chip>
            {files.vcpt && <Chip dark>Task · VCPT</Chip>}
            {run.status === "done" && <Chip dark tone="ok">Analysed</Chip>}
            {run.status === "running" && <Chip dark tone="busy">Processing</Chip>}
          </div>
        </div>
      )}

      <div className="relative mt-auto">
        <div className="h-24 opacity-60"><WaveField lines={3} color="#3b6cf6" height={96} amp={12} /></div>
        <div className="space-y-3 px-7 pb-7">
          <div className="flex items-center gap-2 text-xs">
            <span className={`h-2 w-2 rounded-full ${online === null ? "bg-slate-500" : online ? "bg-emerald-400 shadow-[0_0_10px_2px_rgba(52,211,153,0.6)]" : "bg-rose-500"}`} />
            <span className="text-slate-400">{online === null ? "Connecting…" : online ? "Analysis server online" : "Analysis server offline"}</span>
          </div>
          <div className="flex gap-2 text-[11px] leading-relaxed text-slate-500">
            <ShieldAlert size={14} className="mt-0.5 shrink-0" />
            Research use only. Not a diagnostic device.
          </div>
        </div>
      </div>
    </aside>
  );
}

function TopBar({ step, reachable, onGo, online, canReset, onReset }) {
  const s = STEPS[step - 1];
  return (
    <div className="sticky top-0 z-30 border-b border-slate-200/70 bg-[#f3f6fb]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1180px] items-center justify-between gap-4 px-6 py-4 lg:px-12">
        <div className="lg:hidden"><Logo /></div>
        <div className="hidden items-center gap-2 text-sm lg:flex">
          <span className="text-slate-400">Workflow</span>
          <span className="text-slate-300">/</span>
          <span className="font-medium text-slate-700">Step {step} · {s.label}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex gap-1 lg:hidden">
            {STEPS.map((x) => (
              <button key={x.id} type="button" disabled={!reachable[x.id]} onClick={() => onGo(x.id)}
                className={`h-2 w-8 rounded-full transition ${x.id === step ? "bg-blue-600" : x.id < step ? "bg-blue-300" : "bg-slate-200"} disabled:opacity-50`}
                aria-label={x.label} />
            ))}
          </div>
          {online === false && <Chip tone="bad">Server offline</Chip>}
          {canReset && (
            <button type="button" onClick={onReset}
              className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50">
              <RotateCcw size={14} /> New analysis
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Chip({ children, dark, tone }) {
  const tones = dark
    ? { ok: "bg-emerald-400/15 text-emerald-300", busy: "bg-blue-400/15 text-blue-300", undefined: "bg-white/[0.07] text-slate-300" }
    : { ok: "bg-emerald-50 text-emerald-700", warn: "bg-amber-50 text-amber-800", bad: "bg-rose-50 text-rose-700",
        blue: "bg-blue-50 text-blue-700", undefined: "bg-slate-100 text-slate-600" };
  return <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-semibold ${tones[tone]}`}>{children}</span>;
}

function StageHeader({ n, eyebrow, title, children, right }) {
  return (
    <div className="fade-up mb-8 flex flex-wrap items-end justify-between gap-6">
      <div className="max-w-2xl">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-600">
          <span className="grid h-5 w-5 place-items-center rounded-md bg-blue-600 text-[10px] text-white">{n}</span>{eyebrow}
        </div>
        <h1 className="mt-3 text-[32px] font-semibold leading-[1.15] tracking-tight text-slate-950">{title}</h1>
        {children && <p className="mt-3 text-[15px] leading-relaxed text-slate-500">{children}</p>}
      </div>
      {right}
    </div>
  );
}

function Card({ className = "", style, children }) {
  return (
    <div style={style}
      className={`rounded-3xl border border-slate-200/70 bg-white p-7 shadow-[0_1px_2px_rgba(15,23,42,0.04),0_12px_32px_-12px_rgba(15,23,42,0.10)] ${className}`}>
      {children}
    </div>
  );
}

function CardHead({ icon: Icon, title, hint, right }) {
  return (
    <div className="mb-6 flex items-start justify-between gap-4">
      <div className="flex items-start gap-3">
        {Icon && (
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-blue-50 text-blue-600">
            <Icon size={17} />
          </span>
        )}
        <div>
          <div className="text-[15px] font-semibold text-slate-900">{title}</div>
          {hint && <div className="mt-0.5 text-xs leading-relaxed text-slate-500">{hint}</div>}
        </div>
      </div>
      {right}
    </div>
  );
}

function PrimaryButton({ children, onClick, disabled, className = "" }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled}
      className={`group inline-flex items-center justify-center gap-2 rounded-2xl bg-linear-to-r from-blue-600 to-indigo-600 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-blue-600/25 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-blue-600/30 disabled:translate-y-0 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none ${className}`}>
      {children}
    </button>
  );
}

/* ====================================================== decorative waves */

function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Periodic pseudo-EEG (integer cycles per panel), so two copies scroll seamlessly.
function wavePath(seed, w, h, amp) {
  const rnd = mulberry32(seed);
  const comps = [[1, 3], [0.55, 9], [0.3, 17], [0.18, 31]].map(([a, f]) => ({
    a, f: f + Math.floor(rnd() * 3), ph: rnd() * Math.PI * 2,
  }));
  let d = "";
  const n = 360;
  for (let i = 0; i <= n; i += 1) {
    const t = i / n;
    const y = comps.reduce((s, c) => s + c.a * Math.sin(2 * Math.PI * c.f * t + c.ph), 0);
    d += `${i ? "L" : "M"}${(t * w).toFixed(1)} ${(h / 2 + (y * amp) / 1.6).toFixed(1)}`;
  }
  return d;
}

function WaveField({ lines = 5, color = "#ffffff", height = 220, amp = 16, opacity = 1 }) {
  const w = 1200;
  const rowH = height / lines;
  const paths = useMemo(() => Array.from({ length: lines }, (_, i) => wavePath(7 + i * 13, w, rowH, amp)),
    [lines, rowH, amp]);
  const panel = (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className="h-full w-1/2">
      {paths.map((d, i) => (
        <path key={i} d={d} transform={`translate(0 ${i * rowH})`} fill="none" stroke={color}
              strokeOpacity={opacity * (0.35 + 0.65 * ((i + 1) / lines))} strokeWidth="1.4" />
      ))}
    </svg>
  );
  return (
    <div className="h-full w-full overflow-hidden" aria-hidden="true">
      <div className="wave-scroll flex h-full w-[200%]">{panel}{panel}</div>
    </div>
  );
}

/* =========================================================== 1. upload */

function UploadStage({ files, setFiles, onRun, online }) {
  return (
    <>
      <div className="fade-up relative overflow-hidden rounded-[28px] bg-[#0b1530] p-10 text-white shadow-2xl shadow-blue-950/20">
        <div className="pointer-events-none absolute inset-0 opacity-50">
          <WaveField lines={6} color="#5b8cff" height={300} amp={18} opacity={0.7} />
        </div>
        <div className="pointer-events-none absolute -right-20 -top-28 h-80 w-80 rounded-full bg-indigo-500/30 blur-3xl" />
        <div className="relative max-w-xl">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-blue-300">
            <Sparkles size={14} /> Step 1 · Input
          </div>
          <h1 className="mt-4 text-[40px] font-semibold leading-[1.1] tracking-tight">
            Analyse a child's EEG
          </h1>
          <p className="mt-4 text-[15px] leading-relaxed text-slate-300">
            Upload a resting-state recording. The pipeline cleans it, turns every 1.5 s of brain activity into an
            image, and scores it with a YOLOv8 classifier, live, on this machine.
          </p>
          <div className="mt-7 flex flex-wrap gap-2">
            {["19 EEG channels", "Wavelet scalograms", "About 20 seconds", "Nothing is stored"].map((t) => (
              <span key={t} className="rounded-full bg-white/10 px-3 py-1.5 text-xs font-medium text-slate-200 ring-1 ring-white/15 backdrop-blur">{t}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-8 grid gap-6 md:grid-cols-2">
        <Dropzone n="01" label="Resting-state recording" tag="EOEC · required" delay={80}
                  hint="Eyes closed, then eyes open, in one .edf file"
                  file={files.eoec} onFile={(f) => setFiles((s) => ({ ...s, eoec: f }))}
                  onClear={() => setFiles((s) => ({ ...s, eoec: null }))} />
        <Dropzone n="02" label="Task recording" tag="VCPT · optional" delay={160}
                  hint="Adds task biomarkers; the prediction uses rest only"
                  file={files.vcpt} onFile={(f) => setFiles((s) => ({ ...s, vcpt: f }))}
                  onClear={() => setFiles((s) => ({ ...s, vcpt: null }))} />
      </div>

      <div className="fade-up mt-6 flex flex-wrap items-center justify-between gap-4 rounded-3xl border border-slate-200/70 bg-white/70 p-5 pl-7 backdrop-blur"
           style={{ animationDelay: "240ms" }}>
        <div className="flex items-center gap-3 text-sm text-slate-500">
          <Info size={16} className="text-slate-400" />
          {online === false ? "The analysis server is offline. Start it with uvicorn first."
            : files.eoec ? "Ready. The analysis runs on this machine and takes about 20 seconds."
              : "Add the resting-state recording to begin."}
        </div>
        <PrimaryButton onClick={onRun} disabled={!files.eoec || online === false}>
          Run analysis <ArrowRight size={16} className="transition group-hover:translate-x-0.5" />
        </PrimaryButton>
      </div>
    </>
  );
}

function Dropzone({ n, label, tag, hint, file, onFile, onClear, delay }) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState(null);
  const input = useRef(null);

  function take(f) {
    if (!f) return;
    if (!/\.edf$/i.test(f.name)) { setError("Please choose an .edf file."); return; }
    setError(null);
    onFile(f);
  }

  const shell = "fade-up relative flex min-h-[260px] flex-col rounded-3xl p-7 transition-all duration-300";
  if (file) {
    return (
      <div className={`${shell} border border-blue-200 bg-white shadow-[0_12px_32px_-12px_rgba(59,108,246,0.25)]`} style={{ animationDelay: `${delay}ms` }}>
        <div className="flex items-start justify-between">
          <span className="text-xs font-semibold tracking-wider text-blue-600">{n}</span>
          <button type="button" onClick={onClear} aria-label="Remove file"
                  className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"><X size={16} /></button>
        </div>
        <div className="mt-auto">
          <div className="relative inline-grid h-14 w-14 place-items-center rounded-2xl bg-linear-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-600/30">
            <FileText size={24} />
            <span className="absolute -bottom-1.5 -right-1.5 grid h-6 w-6 place-items-center rounded-full border-[3px] border-white bg-emerald-500 text-white">
              <Check size={11} strokeWidth={3.5} />
            </span>
          </div>
          <div className="mt-5 text-[17px] font-semibold tracking-tight">{label}</div>
          <div className="mt-1 truncate text-sm text-slate-500" title={file.name}>{file.name}</div>
          <div className="mt-4 flex gap-2"><Chip tone="ok">Loaded</Chip><Chip>{(file.size / 1e6).toFixed(1)} MB</Chip></div>
        </div>
      </div>
    );
  }

  return (
    <div role="button" tabIndex={0} style={{ animationDelay: `${delay}ms` }}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}
      className={`${shell} group cursor-pointer border-2 border-dashed
        ${over ? "scale-[1.015] border-blue-500 bg-blue-50/80 shadow-xl shadow-blue-600/10"
          : "border-slate-300/80 bg-white/60 hover:-translate-y-1 hover:border-blue-400 hover:bg-white hover:shadow-[0_20px_40px_-16px_rgba(59,108,246,0.25)]"}`}>
      <input ref={input} type="file" accept=".edf,.EDF" className="hidden" onChange={(e) => take(e.target.files[0])} />
      <div className="flex items-start justify-between">
        <span className="text-xs font-semibold tracking-wider text-slate-400">{n}</span>
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">{tag}</span>
      </div>
      <div className="mt-auto">
        <span className={`grid h-14 w-14 place-items-center rounded-2xl transition-all duration-300
          ${over ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-500 group-hover:-translate-y-0.5 group-hover:bg-linear-to-br group-hover:from-blue-500 group-hover:to-indigo-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-blue-600/30"}`}>
          <Upload size={22} />
        </span>
        <div className="mt-5 text-[17px] font-semibold tracking-tight">{label}</div>
        <div className="mt-1 text-sm text-slate-500">{hint}</div>
        <div className="mt-4 text-sm text-slate-400">
          Drop an <b className="font-semibold text-slate-600">.edf</b> file or{" "}
          <span className="font-semibold text-blue-600 underline-offset-4 group-hover:underline">browse</span>
        </div>
        {error && <div className="mt-2 text-xs font-medium text-rose-600">{error}</div>}
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
      <StageHeader n={2} eyebrow="Visualisation" title="From brain signal to images"
        right={run.status === "done" && (
          <PrimaryButton onClick={onNext}>See the result <ArrowRight size={16} className="transition group-hover:translate-x-0.5" /></PrimaryButton>
        )}>
        The cleaned EEG is cut into 1.5 s epochs. Each becomes a time–frequency scalogram, the image the classifier
        reads; scalp maps and Grad-CAM show what is in it.
      </StageHeader>

      {run.status === "error" && (
        <Card className="fade-up mb-6 border-rose-200 bg-rose-50/70">
          <div className="flex items-start gap-4">
            <AlertTriangle className="mt-0.5 text-rose-600" size={20} />
            <div className="flex-1">
              <div className="font-semibold text-rose-950">The analysis did not complete</div>
              <div className="mt-1 text-sm text-rose-800">{run.error}</div>
            </div>
            <button type="button" onClick={onBack}
              className="flex items-center gap-2 rounded-xl bg-white px-3.5 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50">
              <RotateCcw size={14} /> Start again
            </button>
          </div>
        </Card>
      )}

      <PipelineCard run={run} elapsed={elapsed} subjectId={subjectId} />

      {running && (
        <div className="mt-6 space-y-6">
          <Card><div className="skeleton h-[380px] rounded-2xl" /></Card>
          <div className="grid grid-cols-4 gap-5">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton aspect-[3/4] rounded-3xl" />)}</div>
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
          <div className="fade-up flex flex-wrap items-center justify-between gap-4 rounded-3xl bg-[#0b1530] p-6 pl-8 text-white" style={{ animationDelay: "320ms" }}>
            <div>
              <div className="font-semibold">{result.n_epochs_analyzed} epochs scored</div>
              <div className="text-sm text-slate-400">Their mean is the child's probability.</div>
            </div>
            <PrimaryButton onClick={onNext}>See the result <ArrowRight size={16} className="transition group-hover:translate-x-0.5" /></PrimaryButton>
          </div>
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
    <Card className="fade-up overflow-hidden !p-0">
      <div className="grid items-stretch md:grid-cols-[260px_1fr]">
        <div className="relative flex flex-col items-center justify-center overflow-hidden bg-[#0b1530] p-8 text-white">
          <div className="pointer-events-none absolute inset-0 opacity-30"><WaveField lines={4} color="#5b8cff" height={200} amp={14} /></div>
          <div className="relative grid h-28 w-28 place-items-center">
            <svg viewBox="0 0 120 120" className={`absolute inset-0 ${running ? "spin-slow" : ""}`}>
              <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="6" />
              <circle cx="60" cy="60" r="52" fill="none" stroke="url(#ring)" strokeWidth="6" strokeLinecap="round"
                      pathLength="100" strokeDasharray={done ? "100 100" : "28 100"} transform="rotate(-90 60 60)" />
              <defs><linearGradient id="ring" x1="0" x2="1"><stop offset="0" stopColor="#60a5fa" /><stop offset="1" stopColor="#818cf8" /></linearGradient></defs>
            </svg>
            <div className="relative text-center">
              {done ? <Check size={34} strokeWidth={2.5} className="mx-auto text-emerald-300" />
                : <div className="text-3xl font-semibold tabular-nums">{elapsed.toFixed(0)}<span className="text-base text-slate-400">s</span></div>}
            </div>
          </div>
          <div className="relative mt-5 text-center">
            <div className="text-sm font-semibold">{done ? "Analysis complete" : run.status === "error" ? "Stopped" : "Analysing"}</div>
            <div className="mt-0.5 text-xs text-slate-400">{subjectId || "Recording"}</div>
          </div>
        </div>
        <div className="p-8">
          <div className="mb-6 flex items-center justify-between">
            <div className="text-[15px] font-semibold">Processing pipeline</div>
            <Chip tone={done ? "ok" : run.status === "error" ? "bad" : "blue"}>
              {done ? `${run.result.n_epochs_analyzed} epochs classified` : run.status === "error" ? "Error" : "Typical timing"}
            </Chip>
          </div>
          <ol className="grid grid-cols-3 gap-x-4 gap-y-6 xl:grid-cols-6">
            {PIPELINE.map((p, i) => {
              const state = done ? "done" : !running ? "idle" : i < activeIdx ? "done" : i === activeIdx ? "active" : "idle";
              return (
                <li key={p.label} className="relative">
                  {i < PIPELINE.length - 1 && (
                    <span className="absolute left-10 right-0 top-[17px] hidden h-0.5 rounded bg-slate-100 xl:block">
                      <span className={`block h-full rounded bg-linear-to-r from-blue-500 to-indigo-500 transition-all duration-700 ${state === "done" ? "w-full" : "w-0"}`} />
                    </span>
                  )}
                  <span className={`relative grid h-9 w-9 place-items-center rounded-xl text-xs font-semibold transition-all duration-500
                    ${state === "done" ? "bg-linear-to-br from-blue-500 to-indigo-600 text-white"
                      : state === "active" ? "pulse-ring bg-blue-50 text-blue-600 ring-2 ring-blue-500" : "bg-slate-100 text-slate-400"}`}>
                    {state === "done" ? <Check size={15} strokeWidth={2.5} /> : i + 1}
                  </span>
                  <div className={`mt-3 text-sm font-semibold ${state === "idle" ? "text-slate-400" : "text-slate-900"}`}>{p.label}</div>
                  <div className="text-xs text-slate-400">{p.detail}</div>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </Card>
  );
}

function SectionLabel({ n, title, hint }) {
  return (
    <div className="mb-5 flex items-baseline gap-3">
      <span className="text-sm font-semibold tabular-nums text-blue-600">{n}</span>
      <div>
        <span className="text-[17px] font-semibold tracking-tight">{title}</span>
        {hint && <span className="ml-3 text-sm text-slate-400">{hint}</span>}
      </div>
    </div>
  );
}

function Toggle({ value, onChange, options }) {
  return (
    <div className="inline-flex rounded-xl bg-slate-100 p-1">
      {options.map(([v, label]) => (
        <button key={v} type="button" onClick={() => onChange(v)}
          className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all
            ${value === v ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}>
          {label}
        </button>
      ))}
    </div>
  );
}

function SignalSection({ result }) {
  const tasks = Object.keys(result.signal || {});
  const [task, setTask] = useState(tasks[0] || "EC");
  const s = result.signal?.[task];
  if (!s) return null;
  return (
    <Card className="fade-up">
      <div className="mb-2 flex flex-wrap items-start justify-between gap-4">
        <SectionLabel n="01" title="Cleaned EEG signal"
          hint={`${s.channels.length} channels · ${s.n_epochs} epochs of 1.5 s · after filtering and ICA`} />
        <Toggle value={task} onChange={setTask} options={tasks.map((t) => [t, CONDITION[t]])} />
      </div>
      <EEGViewer key={task} signal={s} />
    </Card>
  );
}

function EEGViewer({ signal }) {
  const W = 1040, rowH = 24, gutter = 46, top = 6;
  const n = signal.uv[0].length;
  const H = top + signal.channels.length * rowH + 26;
  const scale = useMemo(() => {
    const all = signal.uv.flat().map(Math.abs).sort((a, b) => a - b);
    const p95 = all[Math.floor(all.length * 0.95)] || 1;
    return (rowH * 0.48) / p95;
  }, [signal]);
  const x = (i) => gutter + (i / (n - 1)) * (W - gutter - 8);
  const perEpoch = n / signal.n_epochs;
  const clamp = rowH * 0.62;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="EEG signal">
      {signal.channels.map((ch, r) => (
        <rect key={`bg${ch}`} x={gutter} y={top + r * rowH} width={W - gutter - 8} height={rowH}
              fill={r % 2 ? "#f8fafc" : "#ffffff"} />
      ))}
      {Array.from({ length: signal.n_epochs + 1 }, (_, k) => (
        <line key={`e${k}`} x1={x(Math.min(n - 1, k * perEpoch))} x2={x(Math.min(n - 1, k * perEpoch))}
              y1={top} y2={top + signal.channels.length * rowH} stroke="#e2e8f0" strokeDasharray="3 4" />
      ))}
      {signal.channels.map((ch, r) => {
        const mid = top + r * rowH + rowH / 2;
        let d = "";
        signal.uv[r].forEach((v, i) => {
          const y = mid - Math.max(-clamp, Math.min(clamp, v * scale));
          d += `${i ? "L" : "M"}${x(i).toFixed(1)} ${y.toFixed(1)}`;
        });
        return (
          <g key={ch}>
            <text x={gutter - 10} y={mid + 3.5} fontSize="10.5" fontWeight="600" fill="#64748b" textAnchor="end">{ch}</text>
            <path d={d} pathLength="1" className="draw" style={{ animationDelay: `${r * 45}ms` }}
                  fill="none" stroke={r % 2 ? "#3b6cf6" : "#1e3a8a"} strokeWidth="1" strokeLinejoin="round" />
          </g>
        );
      })}
      {Array.from({ length: signal.n_epochs + 1 }, (_, k) => (
        <text key={`t${k}`} x={x(Math.min(n - 1, k * perEpoch))} y={H - 8} fontSize="10.5" fill="#94a3b8" textAnchor="middle">
          {(k * 1.5).toFixed(1)} s
        </text>
      ))}
    </svg>
  );
}

function MiniTrace({ uv, delay = 0 }) {
  const W = 300, rowH = 15;
  const n = uv[0].length;
  const all = uv.flat().map(Math.abs).sort((a, b) => a - b);
  const scale = (rowH * 0.5) / (all[Math.floor(all.length * 0.95)] || 1);
  return (
    <svg viewBox={`0 0 ${W} ${uv.length * rowH + 4}`} className="w-full" aria-hidden="true">
      {uv.map((row, r) => {
        const mid = 2 + r * rowH + rowH / 2;
        const d = row.map((v, i) => `${i ? "L" : "M"}${((i / (n - 1)) * W).toFixed(1)} ${(mid - Math.max(-rowH * 0.6, Math.min(rowH * 0.6, v * scale))).toFixed(1)}`).join("");
        return <path key={r} d={d} pathLength="1" className="draw" style={{ animationDelay: `${delay + r * 60}ms` }}
                     fill="none" stroke="#3b6cf6" strokeOpacity={0.55 + 0.45 * (r / uv.length)} strokeWidth="1.1" />;
      })}
    </svg>
  );
}

function ScalogramSection({ result }) {
  const [task, setTask] = useState("EC");
  const shown = result.images.scalograms.filter((s) => s.task === task);
  const channels = result.images.scalogram_channels || [];
  return (
    <Card className="fade-up" style={{ animationDelay: "120ms" }}>
      <div className="mb-2 flex flex-wrap items-start justify-between gap-4">
        <SectionLabel n="02" title="Signal → time–frequency image" hint="Continuous wavelet transform of each epoch" />
        <Toggle value={task} onChange={setTask} options={[["EC", "Eyes closed"], ["EO", "Eyes open"]]} />
      </div>
      <div key={task} className="grid grid-cols-2 gap-5 xl:grid-cols-4">
        {shown.map((s, i) => (
          <figure key={s.epoch} className="fade-up group rounded-2xl border border-slate-200/70 bg-slate-50/60 p-3 transition hover:-translate-y-1 hover:bg-white hover:shadow-lg"
                  style={{ animationDelay: `${i * 90}ms` }}>
            <div className="rounded-xl bg-white px-2 py-2 ring-1 ring-slate-100">
              {s.signal_uv && <MiniTrace uv={s.signal_uv} delay={i * 90} />}
            </div>
            <div className="flex items-center justify-center gap-1.5 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              <ArrowDown size={12} /> wavelet
            </div>
            <img src={png(s.png_base64)} alt={`Scalogram, ${CONDITION[task]}, epoch ${s.epoch}`}
                 className="aspect-square w-full rounded-xl object-cover ring-1 ring-slate-200" />
            <figcaption className="mt-3 flex items-center justify-between px-1">
              <span className="text-xs font-medium text-slate-500">Epoch {s.epoch}</span>
              <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold tabular-nums ${s.p_adhd >= 0.5 ? "bg-blue-50 text-blue-700" : "bg-slate-100 text-slate-600"}`}>
                p {s.p_adhd.toFixed(2)}
              </span>
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="mt-5 text-xs leading-relaxed text-slate-400">
        Channels {channels.join(", ")} stacked top to bottom; in each, 0.5–50 Hz from bottom to top and time left to right.
        Red: absolute power · green: power above the 1/f background · blue: power relative to each frequency.
        p is the classifier's ADHD probability for that single epoch.
      </p>
    </Card>
  );
}

function TopomapSection({ result }) {
  return (
    <Card className="fade-up" style={{ animationDelay: "200ms" }}>
      <SectionLabel n="03" title="Scalp maps" hint="Band power across the head" />
      <div className="grid grid-cols-2 gap-5">
        {result.images.topomaps.map((t) => (
          <figure key={t.task}>
            <img src={png(t.png_base64)} alt={`Topographic map, ${CONDITION[t.task]}`}
                 className="aspect-square w-full rounded-2xl bg-white object-contain p-2 ring-1 ring-slate-200" />
            <figcaption className="mt-3 flex items-center justify-between text-sm">
              <span className="font-semibold">{CONDITION[t.task]}</span>
              <span className="text-xs text-slate-400">epoch {t.epoch}</span>
            </figcaption>
          </figure>
        ))}
      </div>
      <div className="mt-5 flex flex-wrap gap-1.5">
        {["δ delta", "θ theta", "α alpha", "β beta", "γ gamma"].map((b) => <Chip key={b}>{b}</Chip>)}
      </div>
    </Card>
  );
}

function GradCamSection({ result }) {
  return (
    <Card className="fade-up" style={{ animationDelay: "260ms" }}>
      <SectionLabel n="04" title="Where the model looked" hint="Grad-CAM" />
      <img src={png(result.gradcam_overlay_png_base64)} alt="Grad-CAM overlay"
           className="aspect-square w-full rounded-2xl object-cover ring-1 ring-slate-200" />
      <p className="mt-4 text-xs leading-relaxed text-slate-500">
        Warm regions of this scalogram drove the classifier's score most. It describes the model, not the child:
        a way to inspect what the network uses, not proof that the pattern is physiological.
      </p>
    </Card>
  );
}

/* ========================================================== 3. results */

function ResultsStage({ result, reference, subjectId, onStudy }) {
  const qc = result.quality_control;
  return (
    <>
      <StageHeader n={3} eyebrow="Results" title={`Analysis of ${subjectId || "this recording"}`}>
        The classifier's score, where it falls among children the model has never seen, and the classical EEG
        biomarkers measured from the same recording.
      </StageHeader>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.4fr]">
        <ScoreCard result={result} />
        <Card className="fade-up" style={{ animationDelay: "100ms" }}>
          <CardHead icon={BarChart3} title="Reference distribution"
            hint={`Among ${reference.length} children the model never trained on, scored by this same pipeline.`} />
          <ReferenceStrip reference={reference} value={result.p_adhd} subjectId={subjectId} />
        </Card>
      </div>

      <div className="mt-6 grid gap-6 sm:grid-cols-2 xl:grid-cols-5">
        {BIOMARKERS.map((b, i) => <BiomarkerTile key={b.key} b={b} values={result.classical_biomarkers} delay={160 + i * 60} />)}
        <ReactivityTile value={result.classical_biomarkers.alpha_reactivity} delay={400} />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Card className="fade-up" style={{ animationDelay: "300ms" }}>
          <CardHead icon={Activity} title="Epoch-by-epoch scores"
            hint="The classifier's score for every classified epoch; the child's score is their mean." />
          <EpochHistogram epochs={result.epoch_probabilities} mean={result.p_adhd} />
        </Card>
        {qc && <QualityCard qc={qc} />}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="fade-up flex gap-4 rounded-3xl border border-amber-200/80 bg-linear-to-br from-amber-50 to-orange-50/40 p-7" style={{ animationDelay: "380ms" }}>
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-100 text-amber-700"><ShieldAlert size={20} /></span>
          <div>
            <div className="font-semibold text-amber-950">This is a decision-support research tool, not a diagnostic device.</div>
            <p className="mt-1.5 text-sm leading-relaxed text-amber-900/75">{result.disclaimer}</p>
          </div>
        </div>
        <button type="button" onClick={onStudy}
          className="fade-up group relative overflow-hidden rounded-3xl bg-[#0b1530] p-7 text-left text-white transition hover:-translate-y-1 hover:shadow-2xl hover:shadow-blue-950/30"
          style={{ animationDelay: "440ms" }}>
          <div className="pointer-events-none absolute inset-0 opacity-25"><WaveField lines={3} color="#5b8cff" height={160} amp={12} /></div>
          <div className="relative flex items-center justify-between gap-4">
            <span>
              <span className="block text-xs font-semibold uppercase tracking-[0.16em] text-blue-300">Step 4 · Validation</span>
              <span className="mt-2 block text-lg font-semibold">Does EEG classification hold up?</span>
              <span className="mt-1 block text-sm text-slate-400">The pre-registered study</span>
            </span>
            <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-white/10 transition group-hover:translate-x-1 group-hover:bg-white/20">
              <ArrowRight size={18} />
            </span>
          </div>
        </button>
      </div>
    </>
  );
}

function ScoreCard({ result }) {
  const p = result.p_adhd;
  const v = useCountUp(p);
  const isAdhd = result.predicted_class === "ADHD";
  const R = 92, C = 2 * Math.PI * R, arc = 0.75;
  return (
    <Card className="fade-up flex flex-col items-center">
      <div className="self-start text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Probability of ADHD</div>
      <div className="relative mt-4 h-[230px] w-[230px]">
        <svg viewBox="0 0 230 230" className="absolute inset-0">
          <defs>
            <linearGradient id="gauge" x1="0" y1="1" x2="1" y2="0">
              <stop offset="0" stopColor="#60a5fa" /><stop offset="1" stopColor="#4f46e5" />
            </linearGradient>
          </defs>
          <circle cx="115" cy="115" r={R} fill="none" stroke="#eef2f7" strokeWidth="16" strokeLinecap="round"
                  strokeDasharray={`${C * arc} ${C}`} transform="rotate(135 115 115)" />
          <circle cx="115" cy="115" r={R} fill="none" stroke="url(#gauge)" strokeWidth="16" strokeLinecap="round"
                  strokeDasharray={`${C * arc * v} ${C}`} transform="rotate(135 115 115)" />
          {/* the 0.5 decision threshold sits at the top of the 270-degree arc */}
          <line x1="115" y1="8" x2="115" y2="30" stroke="#0b1530" strokeWidth="2" strokeOpacity="0.35" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-[64px] font-semibold leading-none tracking-tight tabular-nums text-slate-950">{v.toFixed(2)}</div>
          <span className={`mt-3 rounded-full px-3 py-1 text-xs font-semibold ${isAdhd ? "bg-linear-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-600/30" : "bg-slate-100 text-slate-700"}`}>
            {isAdhd ? "ADHD pattern" : "Control pattern"}
          </span>
        </div>
        <span className="absolute bottom-3 left-7 text-[11px] font-medium text-slate-400">0</span>
        <span className="absolute bottom-3 right-7 text-[11px] font-medium text-slate-400">1</span>
      </div>
      <dl className="mt-4 grid w-full grid-cols-3 gap-3 border-t border-slate-100 pt-5 text-center">
        {[["Epochs", result.n_epochs_analyzed], ["Confidence", pct(result.confidence)], ["Threshold", "0.50"]].map(([k, val]) => (
          <div key={k}><dt className="text-[11px] font-medium text-slate-400">{k}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{val}</dd></div>
        ))}
      </dl>
    </Card>
  );
}

function ReferenceStrip({ reference, value, subjectId }) {
  const W = 660, L = 78, R = 16, ROWS = [{ group: "ADHD", y: 70 }, { group: "Control", y: 158 }];
  const x = (v) => L + v * (W - L - R);
  const lanes = [0, -11, 11, -22, 22];

  // Stack dots that would overlap into lanes, so every child stays visible.
  const placed = useMemo(() => ROWS.flatMap((row) => {
    const taken = lanes.map(() => []);
    return reference.filter((r) => r.group === row.group).sort((a, b) => a.p_adhd - b.p_adhd).map((r) => {
      let lane = taken.findIndex((xs) => xs.every((u) => Math.abs(u - r.p_adhd) > 0.022));
      if (lane < 0) lane = 0;
      taken[lane].push(r.p_adhd);
      return { ...r, cx: x(r.p_adhd), cy: row.y + lanes[lane] };
    });
  }), [reference]); // eslint-disable-line react-hooks/exhaustive-deps

  // A child who is in the reference set is not compared with themself.
  const others = reference.filter((r) => r.subject_id !== subjectId);
  const above = (g) => others.filter((r) => r.group === g && r.p_adhd >= value).length;
  const total = (g) => others.filter((r) => r.group === g).length;
  const labelX = Math.min(Math.max(x(value), L + 70), W - R - 70);

  return (
    <div>
      <svg viewBox={`0 0 ${W} 222`} className="w-full" role="img" aria-label="Reference distribution">
        {ROWS.map((row) => (
          <g key={row.group}>
            <rect x={L - 8} y={row.y - 30} width={W - L - R + 16} height={60} rx={30} fill="#f5f7fb" />
            <circle cx={8} cy={row.y} r={5} fill={row.group === "ADHD" ? COLORS.adhd : COLORS.control} />
            <text x={20} y={row.y + 4} fontSize="13" fontWeight="600" fill="#334155">{row.group}</text>
          </g>
        ))}
        <line x1={x(0.5)} x2={x(0.5)} y1={34} y2={194} stroke="#cbd5e1" strokeDasharray="4 4" />
        {placed.map((d, i) => {
          const self = subjectId && d.subject_id === subjectId;
          return (
            <circle key={d.subject_id} cx={d.cx} cy={d.cy} r={self ? 8 : 6.5} className="pop"
              style={{ animationDelay: `${200 + i * 22}ms` }}
              fill={d.group === "ADHD" ? COLORS.adhd : COLORS.control} fillOpacity={self ? 1 : 0.8}
              stroke={self ? COLORS.ink : "#fff"} strokeWidth={self ? 3 : 1.5}>
              <title>{`${d.subject_id} · ${d.group} · p ${d.p_adhd.toFixed(2)}`}</title>
            </circle>
          );
        })}
        <line x1={x(value)} x2={x(value)} y1={30} y2={196} stroke={COLORS.ink} strokeWidth="2.5" strokeLinecap="round" />
        <rect x={labelX - 70} y={2} width={140} height={24} rx={12} fill={COLORS.ink} />
        <text x={labelX} y={18} fontSize="12" fontWeight="600" fill="#fff" textAnchor="middle">
          This recording · {value.toFixed(2)}
        </text>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={x(t)} y={216} fontSize="11" fill="#94a3b8" textAnchor="middle">{t}</text>
        ))}
      </svg>
      {reference.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-3">
          {[["ADHD", COLORS.adhd], ["Control", COLORS.control]].map(([g, c]) => (
            <div key={g} className="rounded-2xl bg-slate-50 px-4 py-3">
              <div className="flex items-center gap-2 text-[11px] font-medium text-slate-500">
                <i className="h-2 w-2 rounded-full" style={{ background: c }} />{g} children scoring at or above
              </div>
              <div className="mt-1 text-lg font-semibold tabular-nums">{above(g)} <span className="text-sm font-medium text-slate-400">of {total(g)}</span></div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BiomarkerTile({ b, values, delay }) {
  const ec = values[`${b.key}_ec`], eo = values[`${b.key}_eo`], task = values[`${b.key}_vcpt`];
  const max = Math.max(...[ec, eo, task].filter((v) => v !== null && v !== undefined).map(Math.abs), 1e-9);
  return (
    <Card className="fade-up !p-6" style={{ animationDelay: `${delay}ms` }}>
      <div className="min-h-[2.75rem]">
        <div className="text-sm font-semibold leading-snug text-slate-900">{b.label}</div>
        <div className="text-[11px] text-slate-400">{b.hint}</div>
      </div>
      <div className="mt-3 text-[32px] font-semibold leading-none tracking-tight tabular-nums">{fmt(ec, b.d)}</div>
      <div className="mt-1 text-[11px] font-medium text-slate-400">eyes closed</div>
      <div className="mt-5 space-y-2">
        {[["EC", ec, COLORS.ec], ["EO", eo, COLORS.eo], ["Task", task, "#c7d2fe"]].map(([k, val, c]) => (
          <div key={k} className="flex items-center gap-2 text-[11px]">
            <span className="w-7 font-medium text-slate-400">{k}</span>
            <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
              {val !== null && val !== undefined && (
                <span className="grow-x block h-full rounded-full" style={{ width: `${(Math.abs(val) / max) * 100}%`, background: c, animationDelay: `${delay + 200}ms` }} />
              )}
            </span>
            <span className="w-9 text-right font-semibold tabular-nums text-slate-600">{fmt(val, b.d)}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

function ReactivityTile({ value, delay }) {
  return (
    <Card className="fade-up flex flex-col !p-6" style={{ animationDelay: `${delay}ms` }}>
      <div className="min-h-[2.75rem]">
        <div className="text-sm font-semibold leading-snug text-slate-900">Alpha reactivity</div>
        <div className="text-[11px] text-slate-400">Closed → open</div>
      </div>
      <div className="mt-3 text-[32px] font-semibold leading-none tracking-tight tabular-nums">{fmt(value, 2)}</div>
      <div className="mt-1 text-[11px] font-medium text-slate-400">ratio</div>
      <p className="mt-auto pt-5 text-[11px] leading-relaxed text-slate-400">
        How strongly occipital alpha falls when the eyes open. Values near 1 mean little blocking.
      </p>
    </Card>
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
        <div className="absolute inset-y-0 border-l-2 border-dashed border-slate-300" style={{ left: "50%" }} />
        <div className="absolute -top-1 bottom-0 z-10 w-0.5 rounded bg-[#0b1530]" style={{ left: pct(mean) }}>
          <span className="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-[#0b1530] px-2 py-0.5 text-[10px] font-semibold text-white">
            mean {mean.toFixed(2)}
          </span>
        </div>
        <div className="flex h-full items-end gap-1">
          {counts.map((c, i) => (
            <div key={i} className="grow flex flex-1 flex-col justify-end overflow-hidden rounded-t-md" style={{ height: `${((c.EC + c.EO) / max) * 100}%`, animationDelay: `${i * 30}ms` }}>
              <div style={{ flex: c.EO, background: COLORS.eo }} />
              <div style={{ flex: c.EC, background: COLORS.ec }} />
            </div>
          ))}
        </div>
      </div>
      <div className="mt-2 flex justify-between text-[11px] text-slate-400">
        <span>0 · Control-like</span><span>0.5</span><span>ADHD-like · 1</span>
      </div>
      <div className="mt-4 flex gap-5 text-xs text-slate-500">
        <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-sm" style={{ background: COLORS.ec }} />Eyes closed</span>
        <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-sm" style={{ background: COLORS.eo }} />Eyes open</span>
        <span className="flex items-center gap-1.5"><i className="h-3 w-0.5 rounded bg-[#0b1530]" />Mean</span>
      </div>
    </div>
  );
}

function QualityCard({ qc }) {
  const tasks = Object.keys(qc.conditions || {});
  return (
    <Card className="fade-up" style={{ animationDelay: "340ms" }}>
      <CardHead icon={qc.flagged ? AlertTriangle : ShieldCheck} title="Recording quality"
        hint="Share of epochs kept after artifact rejection."
        right={<Chip tone={qc.flagged ? "warn" : "ok"}>{qc.flagged ? "Review advised" : "Passed"}</Chip>} />
      <div className="flex justify-around gap-4">
        {tasks.map((t) => {
          const c = qc.conditions[t];
          const kept = 1 - c.rejection_rate;
          const R = 34, C = 2 * Math.PI * R;
          return (
            <div key={t} className="text-center">
              <div className="relative mx-auto h-[88px] w-[88px]">
                <svg viewBox="0 0 88 88" className="absolute inset-0">
                  <circle cx="44" cy="44" r={R} fill="none" stroke="#eef2f7" strokeWidth="8" />
                  <circle cx="44" cy="44" r={R} fill="none" stroke={c.high_rejection ? "#f59e0b" : "#10b981"} strokeWidth="8"
                          strokeLinecap="round" strokeDasharray={`${C * kept} ${C}`} transform="rotate(-90 44 44)" />
                </svg>
                <div className="absolute inset-0 grid place-items-center text-sm font-semibold tabular-nums">{pct(kept)}</div>
              </div>
              <div className="mt-2 text-xs font-semibold text-slate-700">{CONDITION[t] || t}</div>
              <div className="text-[11px] text-slate-400">{c.n_epochs_surviving}/{c.n_epochs_recorded} kept</div>
            </div>
          );
        })}
      </div>
      {qc.notes?.length > 0 && (
        <details className="mt-6 border-t border-slate-100 pt-4 text-xs text-slate-500">
          <summary className="cursor-pointer font-medium text-slate-600 hover:text-slate-900">
            {qc.notes.length} note{qc.notes.length > 1 ? "s" : ""} from the pipeline
          </summary>
          <ul className="mt-3 space-y-2 leading-relaxed">{qc.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </details>
      )}
    </Card>
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
  if (!h1a) {
    return <Card className="text-sm text-slate-500">The study results could not be loaded from the analysis server.</Card>;
  }
  const order = ["H1a", "H1b", "H3", "H2"].map((h) => study.find((r) => r.hypothesis === h)).filter(Boolean);
  const confirmatory = study.filter((r) => ["H1a", "H1b", "H3"].includes(r.hypothesis));
  const positives = confirmatory.filter((r) => String(r.decision).toLowerCase() === "positive").length;

  return (
    <>
      <StageHeader n={4} eyebrow="Validation" title="The pre-registered study">
        Before the second dataset was opened, the analyses and the pass mark were written down and committed to
        GitHub. These are the results exactly as they came out.
      </StageHeader>
      <ValidationHero h1a={h1a} />

      <Card className="fade-up mt-6" style={{ animationDelay: "160ms" }}>
        <CardHead icon={BarChart3} title="All pre-registered results"
          hint={`${positives} of ${confirmatory.length} confirmatory tests positive · dashed line = chance (AUC 0.5)`} />
        <Forest rows={order} />
      </Card>

      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <Card className="fade-up relative overflow-hidden" style={{ animationDelay: "240ms" }}>
          <span className="absolute inset-x-0 top-0 h-1 bg-linear-to-r from-blue-500 to-indigo-600" />
          <CardHead icon={Check} title="What the study shows" />
          <ul className="space-y-3 text-sm leading-relaxed text-slate-600">
            <li>EEG separates children with ADHD from controls within the Nasrabadi cohort, with a model fixed in advance.</li>
            <li>The pipeline detects real differences: every model separates rest from task in the same children.</li>
          </ul>
        </Card>
        <Card className="fade-up relative overflow-hidden" style={{ animationDelay: "300ms" }}>
          <span className="absolute inset-x-0 top-0 h-1 bg-slate-300" />
          <CardHead icon={Info} title="What it does not show" />
          <ul className="space-y-3 text-sm leading-relaxed text-slate-600">
            <li>A marker that holds across cohorts: the model trained on one did not transfer to the other.</li>
            <li>A diagnostic test: an AUC of {h1a.auc.toFixed(2)} is a group difference, far from individual diagnosis.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}

function ValidationHero({ h1a }) {
  const v = useCountUp(h1a.auc, 1400);
  const lo = 0.3, hi = 0.9;
  const at = (u) => `${((u - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="fade-up relative overflow-hidden rounded-[28px] bg-[#0b1530] text-white shadow-2xl shadow-blue-950/20">
      <div className="pointer-events-none absolute inset-0 opacity-35"><WaveField lines={6} color="#5b8cff" height={300} amp={16} opacity={0.6} /></div>
      <div className="pointer-events-none absolute -left-24 -top-24 h-80 w-80 rounded-full bg-blue-600/30 blur-3xl" />
      <div className="relative grid gap-10 p-10 lg:grid-cols-[1.1fr_1fr]">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full bg-emerald-400/15 px-3 py-1.5 text-xs font-semibold text-emerald-300 ring-1 ring-emerald-400/30">
            <ShieldCheck size={14} /> Pre-registered positive · H1a
          </div>
          <div className="mt-6 text-[88px] font-semibold leading-none tracking-tight tabular-nums">{v.toFixed(3)}</div>
          <div className="mt-3 text-slate-300">AUC · ADHD versus Control · {h1a.n} children in an independent cohort</div>

          <div className="mt-8 max-w-md">
            <div className="relative h-10">
              <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10" />
              <div className="absolute top-0 bottom-0 border-l-2 border-dashed border-amber-300/80" style={{ left: at(0.5) }}>
                <span className="absolute -top-5 -translate-x-1/2 text-[10px] font-semibold text-amber-300">chance</span>
              </div>
              <div className="grow-x absolute top-1/2 h-2 -translate-y-1/2 rounded-full bg-linear-to-r from-blue-400 to-indigo-400"
                   style={{ left: at(h1a.ci_low), width: `calc(${at(h1a.ci_high)} - ${at(h1a.ci_low)})`, animationDelay: "500ms" }} />
              <div className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-4 border-[#0b1530] bg-white shadow-[0_0_16px_4px_rgba(96,165,250,0.6)]"
                   style={{ left: at(h1a.auc) }} />
            </div>
            <div className="mt-1 flex justify-between text-[11px] text-slate-500">
              {[0.3, 0.5, 0.7, 0.9].map((t) => <span key={t}>{t.toFixed(1)}</span>)}
            </div>
          </div>

          <div className="mt-8 grid grid-cols-3 gap-3">
            {[["95% CI", `${h1a.ci_low.toFixed(3)}–${h1a.ci_high.toFixed(3)}`], ["Permutation p", h1a.perm_p.toFixed(4)], ["Pass mark", "p < 0.0167"]].map(([k, val]) => (
              <div key={k} className="rounded-2xl bg-white/[0.06] px-4 py-3 ring-1 ring-white/10">
                <div className="text-[11px] text-slate-400">{k}</div>
                <div className="mt-1 text-sm font-semibold tabular-nums">{val}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="space-y-3 self-center">
          {[
            ["Independent data", `Nasrabadi et al.: ${h1a.n} children, ${h1a.n_adhd} with ADHD. Not opened until the plan was committed.`],
            ["Fixed model", "Logistic regression on 100 spectral features. Nothing tuned after the data was seen."],
            ["Split by child", "No child is ever in both training and testing."],
            ["Strict pass mark", "Bonferroni over three tests, and the whole 95% interval above chance."],
          ].map(([t, d], i) => (
            <div key={t} className="fade-up flex gap-4 rounded-2xl bg-white/[0.05] p-4 ring-1 ring-white/10 backdrop-blur"
                 style={{ animationDelay: `${300 + i * 90}ms` }}>
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-blue-500/20 text-blue-300"><Check size={15} strokeWidth={2.5} /></span>
              <div>
                <div className="text-sm font-semibold">{t}</div>
                <div className="mt-0.5 text-xs leading-relaxed text-slate-400">{d}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Forest({ rows }) {
  const lo = 0.2, hi = 0.9;
  const at = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const grid = "grid grid-cols-[minmax(0,13rem)_1fr_5rem_6.5rem] items-center gap-6";
  return (
    <div>
      {rows.map((r, i) => {
        const decision = String(r.decision).toLowerCase();
        const pos = decision === "positive";
        const est = decision !== "positive" && decision !== "negative";
        return (
          <div key={r.hypothesis} className={`${grid} rounded-2xl px-4 py-4 ${pos ? "bg-linear-to-r from-blue-50 to-indigo-50/40 ring-1 ring-blue-100" : ""}`}>
            <div className="min-w-0">
              <div className="text-sm font-semibold"><span className="mr-2 text-slate-400">{r.hypothesis}</span>{HYPOTHESES[r.hypothesis]?.title}</div>
              <div className="mt-0.5 text-xs leading-relaxed text-slate-500">{HYPOTHESES[r.hypothesis]?.sub}</div>
            </div>
            <div className="relative h-7">
              <div className="absolute -bottom-3 -top-3 border-l-2 border-dashed border-amber-400/80" style={{ left: at(0.5) }} />
              <div className={`grow-x absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full ${pos ? "bg-linear-to-r from-blue-500 to-indigo-600" : est ? "bg-slate-200" : "bg-slate-300"}`}
                   style={{ left: at(r.ci_low), width: `calc(${at(r.ci_high)} - ${at(r.ci_low)})`, animationDelay: `${300 + i * 120}ms` }} />
              <div className={`pop absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white shadow-md ${pos ? "bg-indigo-600" : est ? "bg-slate-300" : "bg-slate-500"}`}
                   style={{ left: at(r.auc), animationDelay: `${600 + i * 120}ms` }} />
            </div>
            <div className="text-sm tabular-nums">
              <div className="font-semibold">{r.auc.toFixed(3)}</div>
              <div className="text-xs text-slate-500">{r.ci_low.toFixed(2)}–{r.ci_high.toFixed(2)}</div>
            </div>
            <span className={`justify-self-start rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide ${pos ? "bg-linear-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-600/25" : "bg-slate-100 text-slate-500"}`}>
              {pos ? "Positive" : est ? "Estimate" : "Negative"}
            </span>
          </div>
        );
      })}
      <div className={`${grid} px-4 pt-1`}>
        <div />
        <div className="relative h-4 text-[11px] text-slate-400">
          {[0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map((t) => <span key={t} className="absolute -translate-x-1/2" style={{ left: at(t) }}>{t.toFixed(1)}</span>)}
        </div>
        <div className="text-[11px] text-slate-400">AUC</div>
        <div />
      </div>
    </div>
  );
}
