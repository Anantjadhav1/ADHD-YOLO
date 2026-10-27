# Frontend

Two dashboards talk to the same FastAPI backend (`backend/app/main.py`).

| | what | how to open |
|---|---|---|
| `app/` | **Presentation dashboard** (React 18, Tailwind CSS 4, Lucide icons, Vite). Four steps: upload an EDF, watch it become scalograms / topomaps / Grad-CAM, read the prediction with biomarkers, quality and the reference distribution, then the pre-registered study. | built once, then served by the backend at `http://localhost:8000/app/` |
| `index.html` | The original single-file dashboard, with the print-to-PDF report. | open the file in a browser |

## Presentation dashboard (`app/`)

Build once (needs Node 18+; works offline afterwards):

```powershell
cd D:\adhd-yolo\frontend\app
npm install
npm run build
```

Run:

```powershell
cd D:\adhd-yolo
py -m uvicorn backend.app.main:app --port 8000
```

Open `http://localhost:8000/app/`. While editing the UI, `npm run dev` in `frontend/app` serves it with hot reload at `http://localhost:5173/app/` and calls the backend on port 8000.

**Demo recording.** Use a child the demo checkpoint never trained or selected on (`fold_0` or `test` in `data_pipeline/splits/subject_splits.csv`). `F10101138` (ADHD, EOEC only) scores 0.65 live and has the mildest quality flag of the correctly classified reference children (38% of eyes-closed epochs rejected, no ICA cap). Every one of them carries some flag, and the page shows it.

Nothing on the page is hard-coded: the prediction, images, biomarkers and quality come from `/predict` for the uploaded file; the reference strip from `/reference` (`docs/reference_scores.csv`); the study from `/study` (`docs/preregistered_results.csv`).
