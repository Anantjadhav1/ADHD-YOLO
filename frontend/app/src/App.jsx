import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, BarChart3, Brain, Check, ChevronLeft, ChevronRight, FileText,
  FlaskConical, Info, Layers, Loader2, RotateCcw, ShieldAlert, ShieldCheck, Upload, X,
} from "lucide-react";

// In development the page runs on Vite's port and talks to uvicorn on :8000; once built,
// the backend serves the page itself at /app/, so the API is the same origin.
const API = import.meta.env.DEV ? "http://localhost:8000" : "";

const STEPS = [
  { id: 1, label: "Upload", hint: "EEG recording", icon: Upload },
  { id: 2, label: "Generate", hint: "EEG to images", icon: Layers },
  { id: 3, label: "Results", hint: "Prediction and biomarkers", icon: BarChart3 },
  { id: 4, label: "Validation", hint: "Pre-registered study", icon: FlaskConical },
];

// The backend reports no progress, so these are the stages it runs, in order, with
// typical start times for one EOEC recording on a laptop CPU. Only the response marks
// the work done.
const PIPELINE = [
  { label: "Load and filter", detail: "0.5–50 Hz band-pass, 50 Hz notch", at: 0 },
  { label: "Remove artifacts", detail: "ICA: eye and muscle components", at: 3 },
  { label: "Segment", detail: "1.5 s epochs, noisy epochs rejected", at: 10 },
  { label: "Build scalograms", detail: "Continuous wavelet transform, 5 channels", at: 13 },
  { label: "Classify", detail: "YOLOv8n-cls, 30 epochs per condition", at: 15 },
  { label: "Explain", detail: "Grad-CAM and topographic maps", at: 21 },
];

const BIOMARKERS = [
  { label: "Theta / beta ratio", hint: "F3, F4, Fz · 4–8 Hz over 12–30 Hz", key: "tbr", d: 2 },
  { label: "Individual alpha frequency", hint: "Hz · peak at O1 / O2", key: "iaf", d: 1 },
  { label: "Aperiodic exponent", hint: "1/f slope, 1–45 Hz", key: "aperiodic_exponent", d: 2 },
  { label: "Frontal alpha asymmetry", hint: "F4 versus F3", key: "frontal_alpha_asym", d: 2 },
];

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v) => `${(v * 100).toFixed(0)}%`;
const png = (b64) => `data:image/png;base64,${b64}`;

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
    <div className="min-h-screen text-slate-900">
      <Header online={online} />
      <main className="mx-auto max-w-6xl px-8 pb-24">
        <Stepper step={step} reachable={reachable} done={run.status === "done"} onGo={setStep} />
        <div key={step} className="fade-up">
          {step === 1 && (
            <UploadStage files={files} setFiles={setFiles} onRun={analyse} online={online} />
          )}
          {step === 2 && (
            <GenerateStage run={run} elapsed={elapsed} onNext={() => setStep(3)} onBack={reset} />
          )}
          {step === 3 && run.result && (
            <ResultsStage result={run.result} reference={reference} subjectId={subjectId}
                          onStudy={() => setStep(4)} onReset={reset} />
          )}
          {step === 4 && <ValidationStage study={study} />}
        </div>
      </main>
      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-8 py-6 text-xs text-slate-500">
          <span>ADHD-YOLO · EEG classification of ADHD in children · thesis research prototype</span>
          <span>Every number on this page comes from the live backend or the committed study files.</span>
        </div>
      </footer>
    </div>
  );
}

/* ----------------------------------------------------------------- frame */

function Header({ online }) {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-8 py-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-blue-600 text-white shadow-sm shadow-blue-600/30">
            <Brain size={22} strokeWidth={1.8} />
          </div>
          <div>
            <div className="text-[15px] font-semibold tracking-tight">ADHD-YOLO</div>
            <div className="text-xs text-slate-500">EEG research workstation</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
            <span className={`h-2 w-2 rounded-full ${online === null ? "bg-slate-300" : online ? "bg-emerald-500" : "bg-rose-500"}`} />
            {online === null ? "Connecting…" : online ? "Analysis server online" : "Analysis server offline"}
          </span>
          <span className="hidden rounded-full bg-slate-900 px-3 py-1.5 text-xs font-medium text-white sm:inline">
            Research use only
          </span>
        </div>
      </div>
    </header>
  );
}

function Stepper({ step, reachable, done, onGo }) {
  return (
    <nav className="my-10 grid grid-cols-4 gap-3">
      {STEPS.map((s, i) => {
        const active = s.id === step;
        const complete = s.id < step || (s.id === 3 && done && step === 4);
        const Icon = s.icon;
        return (
          <button key={s.id} type="button" disabled={!reachable[s.id]} onClick={() => onGo(s.id)}
            className={`group relative flex items-center gap-3 rounded-2xl border px-4 py-3.5 text-left transition-all duration-300
              ${active ? "border-blue-200 bg-white shadow-md shadow-blue-900/5 ring-1 ring-blue-100"
                       : "border-transparent hover:bg-white/70"}
              disabled:cursor-not-allowed disabled:opacity-45`}>
            <span className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl transition-colors
              ${active ? "bg-blue-600 text-white" : complete ? "bg-blue-50 text-blue-600" : "bg-slate-100 text-slate-400"}`}>
              {complete && !active ? <Check size={18} /> : <Icon size={18} />}
            </span>
            <span className="min-w-0">
              <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400">Step {i + 1}</span>
              <span className={`block truncate text-sm font-semibold ${active ? "text-slate-900" : "text-slate-600"}`}>{s.label}</span>
              <span className="block truncate text-xs text-slate-400">{s.hint}</span>
            </span>
          </button>
        );
      })}
    </nav>
  );
}

function StageTitle({ eyebrow, title, children, right }) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-6">
      <div className="max-w-2xl">
        <div className="text-xs font-semibold uppercase tracking-[0.14em] text-blue-600">{eyebrow}</div>
        <h1 className="mt-2 text-[28px] font-semibold leading-tight tracking-tight">{title}</h1>
        {children && <p className="mt-3 text-[15px] leading-relaxed text-slate-500">{children}</p>}
      </div>
      {right}
    </div>
  );
}

function Card({ className = "", children }) {
  return (
    <div className={`rounded-2xl border border-slate-200/80 bg-white p-6 shadow-sm shadow-slate-900/[0.03] ${className}`}>
      {children}
    </div>
  );
}

function CardTitle({ icon: Icon, title, hint }) {
  return (
    <div className="mb-5 flex items-start gap-3">
      {Icon && (
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-slate-100 text-slate-500">
          <Icon size={16} />
        </span>
      )}
      <div>
        <div className="text-[15px] font-semibold">{title}</div>
        {hint && <div className="mt-0.5 text-xs leading-relaxed text-slate-500">{hint}</div>}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- 1. upload */

function UploadStage({ files, setFiles, onRun, online }) {
  return (
    <>
      <StageTitle eyebrow="Step 1 · Input" title="Upload a child's EEG recording">
        Resting-state EEG (eyes closed, then eyes open) is required. The task recording is optional
        and adds task biomarkers. Files are processed on this machine and never stored.
      </StageTitle>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.9fr)]">
        <Dropzone label="Resting-state recording" tag="EOEC · required" file={files.eoec}
                  onFile={(f) => setFiles((s) => ({ ...s, eoec: f }))}
                  onClear={() => setFiles((s) => ({ ...s, eoec: null }))} />
        <Dropzone label="Task recording" tag="VCPT · optional" file={files.vcpt}
                  onFile={(f) => setFiles((s) => ({ ...s, vcpt: f }))}
                  onClear={() => setFiles((s) => ({ ...s, vcpt: null }))} />
        <Card className="flex flex-col">
          <CardTitle icon={Info} title="What happens next" />
          <ol className="space-y-4 text-sm text-slate-600">
            {["Filtering and ICA remove noise, eye and muscle artifact.",
              "The recording is cut into 1.5 s epochs and converted into images.",
              "A YOLOv8 classifier scores each image; scores are averaged per child."].map((t, i) => (
              <li key={t} className="flex gap-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-blue-50 text-xs font-semibold text-blue-600">{i + 1}</span>
                <span className="leading-relaxed">{t}</span>
              </li>
            ))}
          </ol>
          <div className="mt-auto pt-6">
            <button type="button" disabled={!files.eoec || online === false} onClick={onRun}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3.5 text-sm font-semibold text-white shadow-sm shadow-blue-600/30 transition-all hover:-translate-y-0.5 hover:bg-blue-700 hover:shadow-md disabled:translate-y-0 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none">
              Run analysis <ChevronRight size={16} />
            </button>
            <p className="mt-3 text-center text-xs text-slate-400">
              {online === false ? "Start the analysis server first." : "Takes about 30 seconds on a laptop."}
            </p>
          </div>
        </Card>
      </div>
    </>
  );
}

function Dropzone({ label, tag, file, onFile, onClear }) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState(null);
  const input = useRef(null);

  function take(f) {
    if (!f) return;
    if (!/\.edf$/i.test(f.name)) { setError("Please choose an .edf file."); return; }
    setError(null);
    onFile(f);
  }

  if (file) {
    return (
      <div className="fade-up relative flex min-h-[290px] flex-col items-center justify-center rounded-2xl border-2 border-blue-200 bg-white p-8 text-center shadow-sm shadow-blue-900/5">
        <button type="button" onClick={onClear} aria-label="Remove file"
                className="absolute right-4 top-4 rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700">
          <X size={16} />
        </button>
        <span className="relative grid h-14 w-14 place-items-center rounded-2xl bg-blue-50 text-blue-600">
          <FileText size={24} />
          <span className="absolute -bottom-1 -right-1 grid h-6 w-6 place-items-center rounded-full border-2 border-white bg-blue-600 text-white">
            <Check size={12} strokeWidth={3} />
          </span>
        </span>
        <div className="mt-5 text-[15px] font-semibold">{label}</div>
        <div className="mt-1 text-xs font-medium text-slate-400">{tag}</div>
        <div className="mt-5 w-full truncate rounded-lg bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700" title={file.name}>
          {file.name}
        </div>
        <div className="mt-2 text-xs text-slate-500">{(file.size / 1e6).toFixed(1)} MB · ready to analyse</div>
      </div>
    );
  }

  return (
    <div
      role="button" tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}
      className={`group flex min-h-[290px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-300
        ${over ? "scale-[1.01] border-blue-500 bg-blue-50"
               : "border-slate-300 bg-white hover:-translate-y-0.5 hover:border-blue-400 hover:bg-blue-50/40 hover:shadow-lg hover:shadow-blue-900/5"}`}>
      <input ref={input} type="file" accept=".edf,.EDF" className="hidden"
             onChange={(e) => take(e.target.files[0])} />
      <span className={`grid h-14 w-14 place-items-center rounded-2xl transition-all duration-300
        ${over ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-500 group-hover:bg-blue-600 group-hover:text-white"}`}>
        <Upload size={24} />
      </span>
      <div className="mt-5 text-[15px] font-semibold">{label}</div>
      <div className="mt-1 text-xs font-medium text-slate-400">{tag}</div>
      <div className="mt-5 text-sm text-slate-500">
        Drop an <span className="font-medium text-slate-700">.edf</span> file here, or{" "}
        <span className="font-medium text-blue-600 underline-offset-4 group-hover:underline">browse</span>
      </div>
      {error && <div className="mt-3 text-xs font-medium text-rose-600">{error}</div>}
    </div>
  );
}

/* ----------------------------------------------------------- 2. generate */

function GenerateStage({ run, elapsed, onNext, onBack }) {
  const [tab, setTab] = useState("scalograms");
  const result = run.result;
  const running = run.status === "running";

  return (
    <>
      <StageTitle eyebrow="Step 2 · Visualisation" title="From EEG signal to images"
        right={running ? (
          <div className="flex items-center gap-2 rounded-full bg-blue-50 px-4 py-2 text-sm font-medium text-blue-700">
            <Loader2 size={16} className="animate-spin" /> Processing · {elapsed.toFixed(0)} s
          </div>
        ) : run.status === "done" ? (
          <button type="button" onClick={onNext}
            className="flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm shadow-blue-600/30 transition hover:-translate-y-0.5 hover:bg-blue-700">
            View results <ChevronRight size={16} />
          </button>
        ) : null}>
        Each 1.5 s epoch becomes a time–frequency scalogram, the image the classifier reads.
        Topographic maps show band power across the scalp; Grad-CAM shows where the classifier looked.
      </StageTitle>

      {run.status === "error" && (
        <Card className="mb-6 border-rose-200 bg-rose-50/60">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 text-rose-600" size={20} />
            <div className="flex-1">
              <div className="font-semibold text-rose-900">The analysis did not complete</div>
              <div className="mt-1 text-sm text-rose-800">{run.error}</div>
            </div>
            <button type="button" onClick={onBack}
              className="flex items-center gap-2 rounded-lg bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50">
              <RotateCcw size={14} /> Start again
            </button>
          </div>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
        <Card>
          <CardTitle icon={Activity} title="Pipeline" hint={running ? "Typical timing for one recording" : "Completed"} />
          <div className={`mb-6 h-1.5 rounded-full bg-slate-100 ${running ? "indeterminate" : ""}`}>
            {!running && <div className={`h-full rounded-full ${run.status === "error" ? "bg-rose-400" : "bg-blue-600"}`} />}
          </div>
          <ol className="space-y-4">
            {PIPELINE.map((p, i) => {
              const next = PIPELINE[i + 1];
              const state = run.status === "done" ? "done"
                : !running ? "idle"
                : next && elapsed >= next.at ? "done"
                : elapsed >= p.at ? "active" : "idle";
              return (
                <li key={p.label} className="flex gap-3">
                  <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full text-xs transition-colors
                    ${state === "done" ? "bg-blue-600 text-white" : state === "active" ? "bg-blue-50 text-blue-600" : "bg-slate-100 text-slate-400"}`}>
                    {state === "done" ? <Check size={13} /> : state === "active" ? <Loader2 size={13} className="animate-spin" /> : i + 1}
                  </span>
                  <span>
                    <span className={`block text-sm font-medium ${state === "idle" ? "text-slate-400" : "text-slate-800"}`}>{p.label}</span>
                    <span className="block text-xs text-slate-400">{p.detail}</span>
                  </span>
                </li>
              );
            })}
          </ol>
        </Card>

        <Card className="min-h-[520px]">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
            <Segmented value={tab} onChange={setTab} options={[
              ["scalograms", "CWT scalograms"], ["topomaps", "Topographic maps"], ["gradcam", "Grad-CAM"]]} />
            {result && (
              <span className="text-xs text-slate-500">
                {result.n_epochs_analyzed} epochs classified · {result.images.scalograms.length} shown
              </span>
            )}
          </div>
          {tab === "scalograms" && <ScalogramGrid result={result} loading={running} />}
          {tab === "topomaps" && <TopomapGrid result={result} loading={running} />}
          {tab === "gradcam" && <GradCam result={result} loading={running} />}
        </Card>
      </div>
    </>
  );
}

function Segmented({ value, onChange, options }) {
  return (
    <div className="inline-flex rounded-xl bg-slate-100 p-1">
      {options.map(([v, label]) => (
        <button key={v} type="button" onClick={() => onChange(v)}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-all
            ${value === v ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}>
          {label}
        </button>
      ))}
    </div>
  );
}

const CONDITION = { EC: "Eyes closed", EO: "Eyes open" };

function ScalogramGrid({ result, loading }) {
  if (!result) return <SkeletonGrid n={8} cols="grid-cols-4" animate={loading} />;
  return (
    <div className="space-y-6">
      {["EC", "EO"].map((task) => (
        <div key={task}>
          <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">{CONDITION[task]}</div>
          <div className="grid grid-cols-4 gap-4">
            {result.images.scalograms.filter((s) => s.task === task).map((s, i) => (
              <figure key={s.epoch} className="fade-up" style={{ animationDelay: `${i * 70}ms` }}>
                <img src={png(s.png_base64)} alt={`Scalogram, ${CONDITION[task]}, epoch ${s.epoch}`}
                     className="aspect-square w-full rounded-xl object-cover ring-1 ring-slate-200" />
                <figcaption className="mt-2 flex items-center justify-between text-xs text-slate-500">
                  <span>Epoch {s.epoch}</span>
                  <span className="font-medium tabular-nums text-slate-700">p {s.p_adhd.toFixed(2)}</span>
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      ))}
      <p className="text-xs leading-relaxed text-slate-400">
        Five channels (Fz, Cz, Pz, F3, F4) stacked vertically, 0.5–50 Hz from bottom to top, time left to right.
        Red: absolute power · green: power above the 1/f background · blue: power relative to each frequency.
      </p>
    </div>
  );
}

function TopomapGrid({ result, loading }) {
  if (!result) return <SkeletonGrid n={2} cols="grid-cols-2" animate={loading} />;
  return (
    <div>
      <div className="grid grid-cols-2 gap-6">
        {result.images.topomaps.map((t) => (
          <figure key={t.task} className="fade-up">
            <img src={png(t.png_base64)} alt={`Topographic map, ${CONDITION[t.task]}`}
                 className="aspect-square w-full rounded-xl bg-white object-contain p-2 ring-1 ring-slate-200" />
            <figcaption className="mt-3 text-sm font-medium">{CONDITION[t.task]}
              <span className="ml-2 text-xs font-normal text-slate-400">epoch {t.epoch}</span></figcaption>
          </figure>
        ))}
      </div>
      <p className="mt-5 text-xs leading-relaxed text-slate-400">
        One head per band, left to right and top to bottom: delta, theta, alpha, beta, gamma.
        Each map is scaled on its own; warm colours mark the scalp regions with the most power in that band.
      </p>
    </div>
  );
}

function GradCam({ result, loading }) {
  if (!result) return <SkeletonGrid n={1} cols="grid-cols-2" animate={loading} />;
  return (
    <div className="grid items-center gap-8 md:grid-cols-2">
      <img src={png(result.gradcam_overlay_png_base64)} alt="Grad-CAM overlay"
           className="fade-up aspect-square w-full rounded-xl object-cover ring-1 ring-slate-200" />
      <div className="space-y-4 text-sm leading-relaxed text-slate-600">
        <div className="text-[15px] font-semibold text-slate-900">Where the classifier looked</div>
        <p>Grad-CAM highlights the parts of one scalogram that most influenced the classifier's score for the
          predicted class. Warm regions drove the decision.</p>
        <p className="text-xs text-slate-400">
          Shown for the last classified epoch. Attribution describes the model, not the child: it is a way to
          inspect what the network uses, not evidence that the pattern is physiological.
        </p>
      </div>
    </div>
  );
}

function SkeletonGrid({ n, cols, animate }) {
  return (
    <div className={`grid gap-4 ${cols}`}>
      {Array.from({ length: n }, (_, i) => (
        <div key={i} className={`aspect-square rounded-xl ${animate ? "skeleton" : "bg-slate-100"}`} />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------ 3. results */

function ResultsStage({ result, reference, subjectId, onStudy, onReset }) {
  const p = result.p_adhd;
  const isAdhd = result.predicted_class === "ADHD";
  const tasks = Object.keys(result.quality_control?.conditions || {});
  const hasTask = result.classical_biomarkers.tbr_vcpt !== null && result.classical_biomarkers.tbr_vcpt !== undefined;

  return (
    <>
      <StageTitle eyebrow="Step 3 · Results" title={`Analysis of ${subjectId || "this recording"}`}
        right={
          <button type="button" onClick={onReset}
            className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50">
            <RotateCcw size={14} /> New recording
          </button>
        }>
        The classifier's score for this child, where it falls among children the model has never seen,
        and the classical EEG biomarkers measured from the same recording.
      </StageTitle>

      <div className="grid gap-6 lg:grid-cols-[0.85fr_1.5fr]">
        <Card className="flex flex-col">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Probability of ADHD</div>
          <div className="mt-3 flex items-end gap-3">
            <span className="text-6xl font-semibold tabular-nums tracking-tight">{p.toFixed(2)}</span>
            <span className={`mb-2 rounded-full px-3 py-1 text-xs font-semibold
              ${isAdhd ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-700"}`}>
              {isAdhd ? "ADHD pattern" : "Control pattern"}
            </span>
          </div>
          <div className="mt-8">
            <div className="relative h-2.5 rounded-full bg-gradient-to-r from-slate-200 via-slate-200 to-blue-200">
              <div className="absolute top-1/2 h-5 w-px -translate-y-1/2 bg-slate-400" style={{ left: "50%" }} />
              <div className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-4 border-white bg-blue-600 shadow-md"
                   style={{ left: pct(p) }} />
            </div>
            <div className="mt-2 flex justify-between text-[11px] text-slate-400">
              <span>0 · Control</span><span>0.5</span><span>ADHD · 1</span>
            </div>
          </div>
          <dl className="mt-auto grid grid-cols-2 gap-4 border-t border-slate-100 pt-5 text-sm">
            <div><dt className="text-xs text-slate-400">Epochs classified</dt>
              <dd className="mt-0.5 font-medium">{result.n_epochs_analyzed}</dd></div>
            <div><dt className="text-xs text-slate-400">Confidence in class</dt>
              <dd className="mt-0.5 font-medium">{pct(result.confidence)}</dd></div>
          </dl>
        </Card>

        <Card>
          <CardTitle icon={BarChart3} title="Reference distribution"
            hint={`This recording among ${reference.length} children the model never trained on, scored by the same pipeline.`} />
          <ReferenceStrip reference={reference} value={p} subjectId={subjectId} />
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.5fr_0.85fr]">
        <Card>
          <CardTitle icon={Activity} title="Classical EEG biomarkers"
            hint="Computed from the same cleaned recording, independent of the classifier." />
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100 text-left text-xs text-slate-400">
                <th className="pb-3 font-medium">Measure</th>
                <th className="pb-3 text-right font-medium">Eyes closed</th>
                <th className="pb-3 text-right font-medium">Eyes open</th>
                <th className="pb-3 text-right font-medium">Task</th>
              </tr>
            </thead>
            <tbody>
              {BIOMARKERS.map((b) => (
                <tr key={b.key} className="border-b border-slate-100 last:border-0">
                  <td className="py-3.5">
                    <div className="font-medium">{b.label}</div>
                    <div className="text-xs text-slate-400">{b.hint}</div>
                  </td>
                  {["ec", "eo", "vcpt"].map((c) => (
                    <td key={c} className="py-3.5 text-right tabular-nums text-slate-700">
                      {fmt(result.classical_biomarkers[`${b.key}_${c}`], b.d)}
                    </td>
                  ))}
                </tr>
              ))}
              <tr>
                <td className="pt-3.5">
                  <div className="font-medium">Alpha reactivity</div>
                  <div className="text-xs text-slate-400">Alpha drop from eyes closed to eyes open</div>
                </td>
                <td colSpan={3} className="pt-3.5 text-right tabular-nums text-slate-700">
                  {fmt(result.classical_biomarkers.alpha_reactivity, 2)}
                </td>
              </tr>
            </tbody>
          </table>
          {!hasTask && <p className="mt-4 text-xs text-slate-400">No task recording uploaded, so task measures are not available.</p>}
        </Card>

        <QualityCard qc={result.quality_control} tasks={tasks} />
      </div>

      <Card className="mt-6">
        <CardTitle icon={Layers} title="Epoch-by-epoch scores"
          hint="Each tick is one 1.5 s epoch; the child's score is their average." />
        <EpochStrip epochs={result.epoch_probabilities} mean={p} />
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.5fr_0.85fr]">
        <div className="flex gap-4 rounded-2xl border border-amber-200 bg-amber-50/70 p-6">
          <ShieldAlert className="mt-0.5 shrink-0 text-amber-600" size={22} />
          <div>
            <div className="font-semibold text-amber-950">
              This is a decision-support research tool, not a diagnostic device.
            </div>
            <p className="mt-1.5 text-sm leading-relaxed text-amber-900/80">{result.disclaimer}</p>
          </div>
        </div>
        <button type="button" onClick={onStudy}
          className="group flex items-center justify-between gap-4 rounded-2xl bg-slate-900 p-6 text-left text-white transition hover:-translate-y-0.5 hover:shadow-lg">
          <span>
            <span className="block text-xs font-semibold uppercase tracking-wider text-slate-400">Step 4</span>
            <span className="mt-1 block font-semibold">How well does EEG classification hold up?</span>
            <span className="mt-1 block text-sm text-slate-400">The pre-registered validation study</span>
          </span>
          <ChevronRight className="shrink-0 transition group-hover:translate-x-1" />
        </button>
      </div>
    </>
  );
}

function ReferenceStrip({ reference, value, subjectId }) {
  const W = 640, L = 86, R = 18, ROWS = [{ group: "ADHD", y: 62 }, { group: "Control", y: 138 }];
  const x = (v) => L + v * (W - L - R);
  const lanes = [0, -10, 10, -20, 20];

  // Stack dots that would overlap into lanes, so every child stays visible.
  const placed = useMemo(() => ROWS.flatMap((row) => {
    const taken = lanes.map(() => []);
    return reference.filter((r) => r.group === row.group).sort((a, b) => a.p_adhd - b.p_adhd).map((r) => {
      let lane = taken.findIndex((xs) => xs.every((v) => Math.abs(v - r.p_adhd) > 0.022));
      if (lane < 0) lane = 0;
      taken[lane].push(r.p_adhd);
      return { ...r, cx: x(r.p_adhd), cy: row.y + lanes[lane] };
    });
  }), [reference]); // eslint-disable-line react-hooks/exhaustive-deps

  // A child who is in the reference set is not compared with themself.
  const others = reference.filter((r) => r.subject_id !== subjectId);
  const above = (g) => others.filter((r) => r.group === g && r.p_adhd >= value).length;
  const total = (g) => others.filter((r) => r.group === g).length;

  return (
    <div>
      <svg viewBox={`0 0 ${W} 200`} className="w-full" role="img" aria-label="Reference distribution">
        {ROWS.map((row) => (
          <g key={row.group}>
            <line x1={L} x2={W - R} y1={row.y} y2={row.y} stroke="#eef2f6" strokeWidth="26" strokeLinecap="round" />
            <text x={0} y={row.y + 4} fontSize="13" fontWeight="600" fill="#334155">{row.group}</text>
          </g>
        ))}
        <line x1={x(0.5)} x2={x(0.5)} y1={24} y2={170} stroke="#cbd5e1" strokeDasharray="4 4" />
        {placed.map((d) => {
          const self = subjectId && d.subject_id === subjectId;
          return (
            <circle key={d.subject_id} cx={d.cx} cy={d.cy} r={self ? 7.5 : 6}
              fill={d.group === "ADHD" ? "#2563eb" : "#94a3b8"} fillOpacity={self ? 1 : 0.75}
              stroke={self ? "#0f172a" : "#fff"} strokeWidth={self ? 2.5 : 1.5}>
              <title>{`${d.subject_id} · ${d.group} · p ${d.p_adhd.toFixed(2)}`}</title>
            </circle>
          );
        })}
        <line x1={x(value)} x2={x(value)} y1={30} y2={168} stroke="#0f172a" strokeWidth="2" />
        <rect x={Math.min(Math.max(x(value) - 66, L), W - R - 132)} y={0} width={132} height={22} rx={11} fill="#0f172a" />
        <text x={Math.min(Math.max(x(value), L + 66), W - R - 66)} y={15} fontSize="11.5" fontWeight="600" fill="#fff" textAnchor="middle">
          This recording · {value.toFixed(2)}
        </text>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={x(t)} y={192} fontSize="11" fill="#94a3b8" textAnchor="middle">{t}</text>
        ))}
      </svg>
      {reference.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500">
          <span>Other children at or above this score: <b className="font-semibold text-slate-700">{above("ADHD")} of {total("ADHD")}</b> ADHD,{" "}
            <b className="font-semibold text-slate-700">{above("Control")} of {total("Control")}</b> Control</span>
          {subjectId && reference.some((r) => r.subject_id === subjectId) && (
            <span>Outlined dot: this child's own reference score</span>
          )}
        </div>
      )}
    </div>
  );
}

function EpochStrip({ epochs, mean }) {
  const W = 900, L = 96, R = 16;
  const x = (v) => L + v * (W - L - R);
  const rows = [["EC", 26], ["EO", 66]];
  return (
    <svg viewBox={`0 0 ${W} 110`} className="w-full" role="img" aria-label="Epoch scores">
      {rows.map(([task, y]) => (
        <g key={task}>
          <text x={0} y={y + 4} fontSize="12" fill="#64748b">{CONDITION[task]}</text>
          <line x1={L} x2={W - R} y1={y} y2={y} stroke="#eef2f6" strokeWidth="22" strokeLinecap="round" />
          {epochs.filter((e) => e.task === task).map((e, i) => (
            <line key={i} x1={x(e.p_adhd)} x2={x(e.p_adhd)} y1={y - 9} y2={y + 9}
                  stroke="#2563eb" strokeOpacity="0.55" strokeWidth="2" strokeLinecap="round" />
          ))}
        </g>
      ))}
      <line x1={x(0.5)} x2={x(0.5)} y1={10} y2={82} stroke="#cbd5e1" strokeDasharray="4 4" />
      <line x1={x(mean)} x2={x(mean)} y1={8} y2={84} stroke="#0f172a" strokeWidth="2" />
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <text key={t} x={x(t)} y={104} fontSize="11" fill="#94a3b8" textAnchor="middle">{t}</text>
      ))}
    </svg>
  );
}

function QualityCard({ qc, tasks }) {
  if (!qc) return null;
  const names = { EC: "Eyes closed", EO: "Eyes open", VCPT: "Task" };
  return (
    <Card>
      <div className="mb-5 flex items-start justify-between gap-3">
        <CardTitle icon={qc.flagged ? AlertTriangle : ShieldCheck} title="Recording quality"
          hint="Share of epochs removed as too noisy." />
        <span className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold
          ${qc.flagged ? "bg-amber-100 text-amber-800" : "bg-emerald-50 text-emerald-700"}`}>
          {qc.flagged ? "Review advised" : "Passed checks"}
        </span>
      </div>
      <div className="space-y-4">
        {tasks.map((t) => {
          const c = qc.conditions[t];
          return (
            <div key={t}>
              <div className="mb-1.5 flex justify-between text-xs">
                <span className="font-medium text-slate-700">{names[t] || t}</span>
                <span className="tabular-nums text-slate-500">
                  {pct(c.rejection_rate)} removed · {c.n_epochs_surviving}/{c.n_epochs_recorded} kept
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-100">
                <div className={`h-full rounded-full ${c.high_rejection ? "bg-amber-400" : "bg-emerald-400"}`}
                     style={{ width: pct(c.rejection_rate) }} />
              </div>
            </div>
          );
        })}
      </div>
      {qc.notes?.length > 0 && (
        <details className="group mt-5 border-t border-slate-100 pt-4 text-xs text-slate-500">
          <summary className="cursor-pointer list-none font-medium text-slate-600 hover:text-slate-900">
            {qc.notes.length} note{qc.notes.length > 1 ? "s" : ""} from the pipeline
          </summary>
          <ul className="mt-3 space-y-2 leading-relaxed">
            {qc.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        </details>
      )}
    </Card>
  );
}

/* --------------------------------------------------------- 4. validation */

const HYPOTHESES = {
  H1a: { title: "Within an independent cohort", sub: "Nasrabadi et al. · trained and tested by child, 5-fold" },
  H1b: { title: "Transfer between cohorts", sub: "Trained on Rohani et al., tested once on Nasrabadi et al." },
  H3: { title: "A new representation", sub: "Riemannian covariance features · Rohani et al." },
  H2: { title: "Held-out split", sub: "Post-hoc model · Rohani et al. · scored once" },
};

function ValidationStage({ study }) {
  const h1a = study.find((r) => r.hypothesis === "H1a");
  const order = ["H1a", "H1b", "H3", "H2"].map((h) => study.find((r) => r.hypothesis === h)).filter(Boolean);
  const confirmatory = study.filter((r) => ["H1a", "H1b", "H3"].includes(r.hypothesis));
  const positives = confirmatory.filter((r) => String(r.decision).toLowerCase() === "positive").length;

  if (!h1a) {
    return (
      <Card className="text-sm text-slate-500">The study results could not be loaded from the analysis server.</Card>
    );
  }

  return (
    <>
      <StageTitle eyebrow="Step 4 · Validation" title="The pre-registered study">
        Before the second dataset was opened, the analyses and the pass mark were written down and committed
        to GitHub. These are their results, exactly as they came out.
      </StageTitle>

      <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
        <div className="relative overflow-hidden rounded-2xl bg-blue-600 p-8 text-white shadow-lg shadow-blue-600/20">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-blue-100">
            <ShieldCheck size={15} /> Pre-registered positive · H1a
          </div>
          <div className="mt-6 text-7xl font-semibold tabular-nums tracking-tight">{h1a.auc.toFixed(3)}</div>
          <div className="mt-2 text-blue-100">AUC, ADHD versus Control, {h1a.n} children</div>
          <div className="mt-8 grid grid-cols-3 gap-4 border-t border-white/20 pt-6 text-sm">
            <div><div className="text-blue-200">95% CI</div>
              <div className="mt-1 font-semibold tabular-nums">{h1a.ci_low.toFixed(3)}–{h1a.ci_high.toFixed(3)}</div></div>
            <div><div className="text-blue-200">Permutation p</div>
              <div className="mt-1 font-semibold tabular-nums">{h1a.perm_p.toFixed(4)}</div></div>
            <div><div className="text-blue-200">Pass mark</div>
              <div className="mt-1 font-semibold tabular-nums">p &lt; 0.0167</div></div>
          </div>
        </div>
        <Card className="flex flex-col justify-center">
          <CardTitle icon={FlaskConical} title="How it was tested" />
          <ul className="space-y-4 text-sm text-slate-600">
            {[
              ["Independent data", `Nasrabadi et al. (${h1a.n} children, ${h1a.n_adhd} with ADHD), not opened until the plan was committed.`],
              ["Fixed model", "Logistic regression on 100 spectral features; nothing tuned after the data was seen."],
              ["Split by child", "No child is ever in both training and testing."],
              ["Strict pass mark", "Bonferroni over three tests, and the whole 95% interval above 0.5."],
            ].map(([t, d]) => (
              <li key={t} className="flex gap-3">
                <Check size={16} className="mt-0.5 shrink-0 text-blue-600" />
                <span><b className="font-semibold text-slate-800">{t}.</b> {d}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card className="mt-6">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
          <CardTitle icon={BarChart3} title="All pre-registered results"
            hint={`${positives} of ${confirmatory.length} confirmatory tests positive. The dashed line is chance (AUC 0.5).`} />
        </div>
        <Forest rows={order} />
      </Card>

      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <Card className="border-t-4 border-t-blue-600">
          <CardTitle icon={Check} title="What the study shows" />
          <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-600">
            <li>EEG separates children with ADHD from controls within the Nasrabadi cohort, with a model fixed in advance.</li>
            <li>The pipeline finds real differences: every model separates rest from task in the same children.</li>
          </ul>
        </Card>
        <Card className="border-t-4 border-t-slate-300">
          <CardTitle icon={Info} title="What it does not show" />
          <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-600">
            <li>A marker that holds across cohorts: a model trained on one did not transfer to the other.</li>
            <li>A diagnostic test: an AUC of {h1a.auc.toFixed(2)} is a group difference, far from individual diagnosis.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}

function Forest({ rows }) {
  const lo = 0.2, hi = 0.9;
  const at = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div>
      {rows.map((r) => {
        const decision = String(r.decision).toLowerCase();
        const pos = decision === "positive";
        const est = decision !== "positive" && decision !== "negative";
        return (
          <div key={r.hypothesis}
            className={`grid grid-cols-[minmax(0,18rem)_1fr_7rem_7rem] items-center gap-6 rounded-xl px-4 py-4 ${pos ? "bg-blue-50/70" : ""}`}>
            <div className="min-w-0">
              <div className="text-sm font-semibold">
                <span className="mr-2 text-slate-400">{r.hypothesis}</span>{HYPOTHESES[r.hypothesis]?.title}
              </div>
              <div className="mt-0.5 text-xs leading-relaxed text-slate-500">{HYPOTHESES[r.hypothesis]?.sub}</div>
            </div>
            <div className="relative h-6">
              <div className="absolute inset-y-[-10px] border-l-2 border-dashed border-amber-400" style={{ left: at(0.5) }} />
              <div className={`absolute top-1/2 h-1 -translate-y-1/2 rounded-full ${pos ? "bg-blue-600" : est ? "bg-slate-300" : "bg-slate-400"}`}
                   style={{ left: at(r.ci_low), width: `calc(${at(r.ci_high)} - ${at(r.ci_low)})` }} />
              <div className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white shadow
                ${pos ? "bg-blue-600" : est ? "bg-slate-300" : "bg-slate-500"}`} style={{ left: at(r.auc) }} />
            </div>
            <div className="text-sm tabular-nums">
              <div className="font-semibold">{r.auc.toFixed(3)}</div>
              <div className="text-xs text-slate-500">{r.ci_low.toFixed(2)}–{r.ci_high.toFixed(2)}</div>
            </div>
            <span className={`justify-self-start rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide
              ${pos ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-500"}`}>
              {pos ? "Positive" : est ? "Estimate" : "Negative"}
            </span>
          </div>
        );
      })}
      <div className="grid grid-cols-[minmax(0,18rem)_1fr_7rem_7rem] gap-6 px-4 pt-1">
        <div />
        <div className="relative h-4 text-[11px] text-slate-400">
          {[0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map((t) => (
            <span key={t} className="absolute -translate-x-1/2" style={{ left: at(t) }}>{t.toFixed(1)}</span>
          ))}
        </div>
        <div className="text-[11px] text-slate-400">AUC</div>
        <div />
      </div>
    </div>
  );
}
