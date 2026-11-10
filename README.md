<div align="center">

# CLARA

### Complaint-Led Aggregation for Road Anomalies

**Federated learning where the server's aggregation weights are modulated by an urgency signal
extracted from citizen complaints — not by data volume alone.**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Flower 1.29](https://img.shields.io/badge/flower-1.29-ff69b4.svg)](https://flower.ai/)
[![Ultralytics 8.4](https://img.shields.io/badge/ultralytics-8.4-009688.svg)](https://docs.ultralytics.com/)

</div>

---

## The idea in three sentences

Standard federated learning (FedAvg) combines client updates in proportion to how much data each
client holds. On severely Non-IID data that is a poor proxy for how *useful* a client is: a client
with many uninformative images outweighs one with few images that matter.

CLARA adds a second term to the aggregation weights, derived from an **exogenous channel**: an NLP
pipeline reads free-text citizen complaints, infers where each one is about and how urgent it
sounds, and produces a per-client urgency vector.

Crucially, **the text never enters the detector**. There is no multimodal fusion, no text encoder
alongside the vision backbone. The signal acts only on the server-side averaging coefficients, and
the detector architecture is untouched.

```
          complaints (text)                 road images
                 │                                │
        ┌────────▼────────┐              ┌────────▼────────┐
        │  NLP pipeline   │              │  6 FL clients   │
        │   (4 stages)    │              │    YOLOv8n      │
        └────────┬────────┘              └────────┬────────┘
                 │ urgency vector                 │ weights + n_k
                 └──────────────┬─────────────────┘
                        ┌───────▼───────┐
                        │   FL server   │
                        │ NLP-Weighted  │
                        └───────────────┘
```

---

## Results

Primary metric is `critical_ap50` — AP@IoU 0.5 computed **on the critical class D40 (potholes)
only**. Figures are mean ± std over three seeds at federated round 10.

| Strategy | `critical_ap50` | mAP@50 | mAP@50-95 |
|---|---|---|---|
| FedAvg (baseline) | 0.091 ± 0.002 | 0.145 ± 0.010 | 0.058 ± 0.004 |
| Oracle-Weighted | 0.097 ± 0.010 | 0.143 ± 0.005 | 0.056 ± 0.001 |
| **NLP-Weighted** | **0.105 ± 0.005** | 0.145 ± 0.005 | 0.057 ± 0.003 |

**+15 % relative on the critical class, with general metrics statistically unchanged.** The
mechanism does not add capacity — it reallocates it toward the class of interest without measurable
degradation elsewhere.

Two further findings:

- **Robustness plateau.** With a synthetically degraded signal, performance is flat for simulated
  NLP accuracy between 0.7 and 1.0. The method tolerates a 30 % error rate with no measurable loss.
- **End-to-end confirmation.** Running the *real* NLP pipeline as the signal source yields 0.104,
  essentially reproducing the oracle-driven result. The mechanism needs a correct *ranking* of
  clients, not a precise vector.

### NLP module, stage by stage

| Stage | Module | What it does | Accuracy |
|---|---|---|---|
| 1 | `ner_module.py` | extracts the locating entity (spaCy `en_core_web_lg`) | 91.7 % |
| 2 | `location_mapper.py` | hybrid lookup + raw-text fallback → federated client | 100 % |
| 3 | `sentiment_module.py` | Twitter-RoBERTa + lexical heuristics → severity in [0, 1] | 86.7 % |
| 4 | `aggregator.py` | sum per client, floor at `u_min`, normalise to 1 | — |

Because every stage has its own ground truth, errors are **attributable**: when the end-to-end
vector drifts, you can say which stage caused it.

---

## The methodological problem

Evaluating an NLP-informed mechanism with an NLP-derived metric is circular: the method looks
superior because it optimises exactly what you chose to measure. Three disciplines break the loop.

<table>
<tr><td width="33%" valign="top">

**A · Exogenous criticality**

The critical class is fixed *before* looking at any output, on a disciplinary criterion external to
the system: potholes are a priority structural defect in pavement-engineering literature. No result
can change it.

</td><td width="33%" valign="top">

**B · Informational upper bound**

`OracleUrgencyProvider` reads the true class distribution from the dataset manifest, simulating a
perfect NLP module. It caps expectations honestly: no real pipeline can beat it by construction.

</td><td width="33%" valign="top">

**C · Controlled noise sweep**

`noise_injection.py` interpolates between the oracle vector and a random permutation, letting you
*dial* the signal quality — impossible with a real pipeline, whose error rate is fixed.

</td></tr>
</table>

A consequence worth stating up front: **the main sweep deliberately uses synthetic providers, not
the real pipeline.** The real pipeline has a distinct job — proving the system works end to end —
and gets its own run.

---

## Repository layout

```
├── configs/
│   ├── base.yaml                          # shared defaults
│   ├── data/rdd2022_country_split.yaml    # the six federated clients
│   ├── experiments/                       # one file per research question
│   │   ├── rq1_baselines_vs_proposed.yaml
│   │   ├── rq2_nlp_noise_sweep.yaml
│   │   └── ablation_cap_decay.yaml
│   └── strategies/                        # one file per aggregation strategy
│       ├── fedavg.yaml
│       ├── oracle_weighted.yaml
│       └── nlp_weighted.yaml
├── src/
│   ├── fl_server/
│   │   ├── custom_strategies.py           # Oracle-Weighted, NLP-Weighted
│   │   └── server.py
│   ├── fl_client/
│   │   ├── client.py                      # Flower client
│   │   └── local_training.py              # wraps the Ultralytics trainer
│   ├── nlp/
│   │   ├── ner_module.py                  # stage 1
│   │   ├── location_mapper.py             # stage 2
│   │   ├── sentiment_module.py            # stage 3
│   │   ├── aggregator.py                  # stage 4
│   │   ├── urgency_provider.py            # the FL ↔ NLP interface
│   │   └── noise_injection.py             # controlled degradation (component C)
│   ├── models/
│   │   ├── yolo_wrapper.py
│   │   └── nlp_pipeline.py                # the four stages, end to end
│   ├── evaluation/critical_classes.py     # critical_ap50
│   └── utils/{metrics,time_decay,risk_matrix}.py
├── scripts/
│   ├── convert_voc_to_yolo.py             # RDD2022: Pascal VOC → YOLO, per client
│   ├── run_full_pipeline.py               # orchestrates the whole experimental sweep
│   ├── aggregate_pipeline_results.py      # results → CSV, figures, LaTeX tables
│   ├── eval_sentiment.py                  # NLP module in isolation
│   └── analyze_noniid_distrib.py          # Non-IID statistics of the partition
├── data/
│   ├── nlp/
│   │   ├── synthetic_complaints.csv            # balanced: 10 complaints × 6 clients
│   │   └── synthetic_complaints_realistic.csv  # proportional to D40 presence
│   └── processed/rdd2022_clients/         # output of the converter
├── experiments/report_pipeline/<run_id>/  # one directory per sweep
└── docs/main.pdf                          # the full report
```

### The interface that makes the protocol possible

The FL and NLP modules communicate through exactly one abstraction, in
`src/nlp/urgency_provider.py`:

```python
class UrgencyProvider(ABC):
    """Source of the per-client, per-round urgency signal."""

    @abstractmethod
    def get_urgency(
        self,
        round_idx: int,
        client_names: list[str],
    ) -> dict[str, float]:
        """Returns {client_name: urgency_score}, normalised to sum to 1."""
```

From the server's point of view the three implementations — oracle, noisy oracle, real pipeline —
are indistinguishable. That is what lets the same aggregation code run against all three with no
branching and no duplicated integration logic.

---

## Setup

```bash
git clone https://github.com/nf0101/CLARA.git
cd CLARA

# with uv (a uv.lock is committed)
uv sync

# or with pip
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m spacy download en_core_web_lg
```

The Twitter-RoBERTa checkpoint is pulled from the Hugging Face Hub on first use and cached
locally — no manual download needed.

### Dataset

RDD2022 is distributed on figshare under DOI
[`10.6084/m9.figshare.21431547`](https://doi.org/10.6084/m9.figshare.21431547). Download it, place
the country archives under `data/raw/`, then convert:

```bash
python scripts/convert_voc_to_yolo.py
```

This produces one self-contained directory per federated client under
`data/processed/rdd2022_clients/`, each with its own `data.yaml` and a `conversion_manifest.json`.
Pointing an Ultralytics command at any of them trains on that country alone.

```bash
python scripts/convert_voc_to_yolo.py --countries Japan India   # subset only
python scripts/convert_voc_to_yolo.py --val-fraction 0.15 --seed 42
```

Two choices baked into the converter:

- Only the four standard classes are kept. The country-specific ones (D43, D44, D50, present almost
  exclusively in the Japanese subset) are discarded so every client shares a uniform label
  vocabulary; discarded counts are logged for transparency.
- `China_Drone` is excluded — drone imagery is a very different visual domain, and the FL literature
  on RDD2022 does not use it. The remaining six follow the convention of Saha et al. (2024).

After conversion you should see roughly:

| Client | Train | Val | D40 instances |
|---|---:|---:|---:|
| `india` | 6 551 | 1 155 | 3 187 |
| `japan` | 8 931 | 1 575 | 2 243 |
| `norway` | 6 937 | 1 224 | 461 |
| `china_motorbike` | 1 681 | 296 | 235 |
| `czech` | 2 405 | 424 | 197 |
| `united_states` | 4 085 | 720 | 135 |

That last column is the whole motivation: **India holds 49 % of all pothole instances while Japan
holds more images overall** — so FedAvg hands Japan the larger weight.

---

## Reproducing the experiments

One command runs everything: RQ1, RQ2, the ablation, the aggregation and the figures.

```bash
python scripts/run_full_pipeline.py                      # default: --scale presentable
```

### Scale presets

| Preset | Seeds | Rounds | Wall clock | Use for |
|---|---:|---:|---|---|
| `debug` | 1 | 3 | ~1 h | checking the environment works |
| `presentable` | 3 | 10 | ~25–30 h | **the configuration reported above** |
| `full` | 5 | 15 | ~360 h | a cluster, not a workstation |

```bash
python scripts/run_full_pipeline.py --scale debug        # quick smoke run
python scripts/run_full_pipeline.py --only rq1           # one phase
python scripts/run_full_pipeline.py --skip ablation      # repeatable
python scripts/run_full_pipeline.py --no-analysis        # skip figures and tables
python scripts/run_full_pipeline.py --resume 20260710_123413
```

### What a run produces

Everything lands under `experiments/report_pipeline/<timestamp>/`:

```
rq1/  rq2/  ablation/        per-run logs and results.json
aggregated.csv               every result, long format
figures/                     rq1_strategy_comparison.png, rq1_per_round.png,
                             rq2_robustness_curve.png, ablation_cap_decay_*.png
tables/                      rq1_summary.tex, rq2_summary.tex, ablation_summary.tex, summary.md
pipeline_manifest.json       the full configuration, for reproducibility
```

Aggregation runs automatically at the end of the sweep. To re-run it on an existing directory —
after tweaking a plot, say:

```bash
python scripts/aggregate_pipeline_results.py \
    --pipeline-dir experiments/report_pipeline/20260710_123413
```

### NLP module in isolation

The pipeline can be evaluated without touching the FL side — useful when iterating on the severity
heuristics, which is where the remaining error lives:

```bash
python scripts/eval_sentiment.py
```

---

## Configuration reference

Defaults live in `configs/base.yaml` and are overridden per experiment under
`configs/experiments/`.

| Parameter | Default | Role |
|---|---|---|
| `alpha` | 0.5 | balance between volume-based and context-based weighting; `1.0` recovers FedAvg |
| `cap` | 0.5 | maximum aggregation weight for any single client |
| `time_decay` (γ) | 1.0 | decay of the urgency signal across rounds; `1.0` means **inactive** |
| `u_min` | 0.01 | floor on raw urgency, so a silent client is never zeroed out |
| `w_T` | 0.4 | contraction applied to the Transformer's deviation from neutral |
| `rounds` | 10 | federated rounds (`presentable` preset) |
| `local_epochs` | 1 | local epochs per round |
| `imgsz` | 416 | input resolution |
| `batch` | 16 | local batch size |

Two notes that matter for interpreting results.

**`time_decay = 1.0` means the decay is off.** The formula reduces to the identity and the signal
keeps full strength at every round. It is varied only in the ablation.

**Neither safeguard ever binds in RQ1 and RQ2.** The highest weight any client reaches is 0.354
against a cap of 0.5, and the decay is inactive. They are dimensioned against a worst case this
dataset does not produce — which is a feature of the result, not a gap: the observed gain cannot be
a hidden regularisation effect.

---

## Hardware and runtime

Developed and run on a single machine: NVIDIA RTX A4000 (16 GB), AMD Threadripper Pro 5955WX
(16 cores), 128 GB RAM, Ubuntu 22.04 under WSL2.

Flower runs in simulation mode on Ray, with `client_resources = {"num_gpus": 0.16}`. That fraction
is a **logical scheduling quota**, letting Ray co-locate all six client actors on the same GPU — it
is not a VRAM partition, and video memory stays shared. Peak GPU utilisation sits around 96–99 %
with roughly 6.5 GB of the 16 GB in use.

A single 10-round run takes about 30 minutes; the full `presentable` sweep about 25 hours.

---

## Known limitations

Stated plainly, because they bound what the numbers mean:

1. **The baseline is FedAvg**, which is known to underperform on severe Non-IID. FedProx and
   SCAFFOLD would likely have narrowed the gap; they are **not implemented** (the YAML stubs under
   `configs/strategies/` are placeholders). Both require intercepting the loss or the gradients
   inside Ultralytics' `DetectionTrainer`, and that coupling is fragile across versions. Note they
   act on *local training* while CLARA acts on *aggregation* — they are composable, not competing.
2. **The complaint dataset is synthetic and small** (60 items, LLM-generated then hand-annotated).
   Its variation covers what the pipeline is designed to handle; it contains no typos, no
   code-switching and no locations outside the mapping table. Treat the accuracies as proof of
   function, not as deployment estimates.
3. **Clients are countries, not intra-city zones**, for want of a public dataset at that granularity.
4. **Three seeds** is the reasonable minimum, not the ideal. Three independent estimates of the same
   nominal configuration span 0.008 — roughly 60 % of the headline effect.
5. **Reduced training scale** (`presentable`), not the scale used by CRDDC leaderboard entries.
6. **The oracle vector is computed over train and validation combined.** The L1 distance from the
   train-only vector is 0.0005, so the effect is negligible, but it is disclosed: it marginally
   strengthens the oracle's character as an upper bound.

---

## Generalising beyond this use case

The interface constrains the *shape* of the signal, not its source. Any process that can emit a
normalised per-client priority vector plugs in unchanged: IoT sensors, weather alerts,
epidemiological data, administrative priorities.

NLP-Weighted is the instance validated here. The family is **Context-Weighted FedAvg**, and
formalising it — characterising the requirements an external signal must satisfy, and what the
validation protocol becomes when component B has no manifest to read — is the extension that
interests me most.

---

## Citation

This repository accompanies a combined Deep Learning + Natural Language Processing project at the
Università degli Studi di Salerno. The full report — validation protocol, complete result tables,
per-stage NLP evaluation — is in [`docs/main.pdf`](docs/main.pdf).

```bibtex
@misc{frugieri2026clara,
  author = {Frugieri, Nicola},
  title  = {{CLARA}: Complaint-Led Aggregation for Road Anomalies},
  year   = {2026},
  note   = {NLP-Weighted Federated Learning for urban road damage detection},
  url    = {https://github.com/nf0101/CLARA}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

RDD2022 is redistributed by its original authors under its own terms; this repository contains no
dataset images.
