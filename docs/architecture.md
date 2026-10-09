# Architecture

This document collects the architecture diagrams for the Kosuri-GPR-Seq2Expr
pipeline. All diagrams are written in [Mermaid](https://mermaid.js.org/) so
they render natively on GitHub without any tooling.

## System overview

```
                                Kosuri-GPR-Seq2Expr
                                =====================

                    +-------------------------------------+
                    |  Data pipeline (src/data)           |
                    |                                     |
   PNAS .xls  --->  |  xls_to_xlsx.py  ---> .xlsx         |
                    |      |                              |
                    |      v                              |
                    |  kosuri_loader.py                   |
                    |   - QC (bad.*, Nopromoter)          |
                    |   - feature extraction (20 cols)     |
                    |      |                              |
                    |      v                              |
                    |  paired_dataset.csv                 |
                    |  (11,476 x 41, 20 features)         |
                    +-------------------------------------+
                                  |
                                  v
                    +-------------------------------------+
                    |  Modelling (src/models,             |
                    |             src/evaluation)         |
                    |                                     |
                    |  run_experiment.py                  |
                    |   - GroupKFold(10) by promoter_id   |
                    |   - GPR (RBF + WhiteKernel)         |
                    |   - RandomForest baseline            |
                    |      |                              |
                    |      v                              |
                    |  final_gpr.joblib                   |
                    |                                     |
                    |  run_nullsette.py                   |
                    |   - 19 element-translocation        |
                    |     virtual mutants (Nullsettes)     |
                    |   - KS + Mann-Whitney tests          |
                    |                                     |
                    |  run_simulate_al.py                  |
                    |   - 3 acquisition functions          |
                    |   - 8 iterations x 150/query         |
                    |   - 3 random seeds                  |
                    +-------------------------------------+
                                  |
                                  v
                    +-------------------------------------+
                    |  Application (src/api, src/ui)      |
                    |                                     |
                    |  FastAPI  :8000   Streamlit :8501    |
                    |   /predict     (5-page dashboard)    |
                    |   /health                           |
                    |   /model/info                       |
                    |   /runs                              |
                    |      |                              |
                    |      v                              |
                    |  Dockerfile + docker-compose.yml     |
                    +-------------------------------------+
```

## Mermaid diagrams (render on GitHub)

### End-to-end system architecture

```mermaid
flowchart TB
    subgraph DL[Data pipeline]
        XLS[".xls<br/>(sd01-03, PNAS)"] -->|xls_to_xlsx.py| XLSX[".xlsx"]
        XLSX -->|kosuri_loader.py<br/>QC + 20 features| CSV["paired_dataset.csv<br/>11,476 rows x 41 cols"]
    end

    subgraph ML[Modelling + evaluation]
        CSV -->|run_experiment.py<br/>GroupKFold by promoter_id| GPR["GPR<br/>(RBF + WhiteKernel)"]
        CSV -->|run_experiment.py| RF[RandomForest]
        GPR --> JOB[final_gpr.joblib]
        GPR -->|run_nullsette.py<br/>19 virtual mutants| OOD["OOD distribution<br/>pred_std +15-76%"]
        CSV -->|run_simulate_al.py| AL["Active learning<br/>3 seeds, 3 acquisitions"]
    end

    subgraph AP[Application]
        JOB --> API["FastAPI<br/>/predict /health /runs"]
        JOB --> UI["Streamlit<br/>5-page dashboard"]
        API --> DOCKER["Dockerfile<br/>+ docker-compose.yml"]
        UI --> DOCKER
    end

    subgraph CI[CI / CD]
        GH[GitHub Actions]
        PY[pytest 7 unit tests]
        GH --> PY
    end

    classDef data fill:#e1f5ff,stroke:#0277bd
    classDef model fill:#fff3e0,stroke:#e65100
    classDef app fill:#e8f5e9,stroke:#2e7d32
    classDef ci fill:#f3e5f5,stroke:#6a1b9a
    class XLS,XLSX,CSV data
    class GPR,RF,JOB,OOD,AL model
    class API,UI,DOCKER app
    class GH,PY ci
```

### Uncertainty evaluation loop

```mermaid
flowchart LR
    subgraph TRAIN["Training set<br/>(GroupKFold, 80%)"]
        K1["Fold 1 train"]
        K2["Fold 2 train"]
        K3["..."]
    end

    subgraph GPR["Trained GPR<br/>(final_gpr.joblib)"]
        GPR_K["kernel: RBF + White<br/>predict mean + std"]
    end

    subgraph TESTS["Evaluation surfaces"]
        N1["Nullsette OOD test<br/>(19 mutants)<br/>-> pred_std up 15-76%"]
        A1["Active learning sim<br/>(3 seeds x 3 acquisitions)<br/>-> marginal / seed-dependent"]
        A2["Held-out test set<br/>(20% stratified by promoter)<br/>-> Test R2, RMSE, Pearson"]
    end

    K1 --> GPR_K
    K2 --> GPR_K
    K3 --> GPR_K
    GPR_K -->|predict mean + std| N1
    GPR_K -->|predict on unlabelled| A1
    GPR_K -->|predict on held-out| A2
```

### Data lineage

```mermaid
flowchart LR
    PNAS[PNAS supplementary<br/>sd01, sd02, sd03 .xls]
    XLSX[sd01.xlsx etc.<br/>11MB total]
    CSV[paired_dataset.csv<br/>11,476 rows]
    RUNS[src/runs/&lt;tag&gt;/<br/>final_gpr.joblib<br/>per-fold metrics<br/>comparison.csv]

    PNAS -->|xls_to_xlsx.py<br/>Excel COM| XLSX
    XLSX -->|kosuri_loader.py<br/>QC + features| CSV
    CSV -->|run_experiment.py| RUNS
```

## Why this shape

* **Strict layering.** `data -> model -> api/ui` is one-directional so we
  can swap out any layer without rewriting the others. The data layer
  never imports from the model layer; the application layer never
  imports from the data layer.
* **Group-aware everywhere.** From the loader (`promoter_id` is the
  group key) to the AL module (test set + initial labelled set are
  derived from a fixed seed per acquisition so strategies are
  comparable) -- the project never accidentally lets the same
  promoter into both train and test.
* **No hard-coded paths.** Every data file, every feature list, every
  GPR hyper-parameter lives under `src/configs/*.json`. The
  application layer (`src/api/main.py`, `src/ui/streamlit_app.py`)
  picks up the latest `__gpr` run automatically.
* **Test-first deliverables.** Every script that produces an
  artefact (`run_experiment.py`, `run_nullsette.py`,
  `run_simulate_al.py`) also writes per-run metrics and a `config
  snapshot` -- the job-application story "I can show exactly what I
  ran" is a property of the codebase, not a manual step.

