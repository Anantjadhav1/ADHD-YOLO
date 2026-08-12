# ADHD-YOLO

YOLO-based classification framework for ADHD diagnosis from EEG/ERP signals, converted to 2D scalograms and topographic heatmaps. See `PROJECT.md` for the full guideline, methodology, and roadmap — read that first.

## Layout

```
adhd-yolo/
├── PROJECT.md          # full guideline — methodology, decisions, roadmap
├── PROGRESS.md         # running log — update this every session
├── data_pipeline/       # Phase 1: MNE preprocessing, CWT/topomap conversion
├── models/              # trained weights, training configs (gitignored — large files)
├── backend/             # FastAPI serving the model (Phase 5)
├── frontend/            # dashboard (Phase 5)
├── notebooks/           # exploratory work — not production code
├── docs/                # Jira epic breakdown, AWS deployment notes
└── docker-compose.yml   # local dev environment, portable to EC2 later
```

## Setup

```bash
git init
git checkout -b main
docker compose build
docker compose up
```

Backend health check: `http://localhost:8000/health`

## Workflow

- `main` is always deployable. Never commit directly to it.
- One branch per feature: `git checkout -b feature/data-conversion`
- Small commits, PR into `main` even solo, delete branch after merge.
- See `docs/jira_board.md` for the epic/story breakdown to enter into Jira.
