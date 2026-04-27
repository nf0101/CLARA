#!/usr/bin/env python3
"""
Setup della struttura di progetto per urban-fl-nlp.

Crea l'albero delle directory, i file __init__.py per i pacchetti Python,
gli stub dei config YAML e i file meta essenziali (.gitignore, README.md,
requirements.txt, pyproject.toml).

Lo script è idempotente: i file gia' esistenti NON vengono sovrascritti.
Viene stampato un riepilogo di cosa e' stato creato e cosa e' stato saltato.

Uso:
    python setup_project.py                    # crea ./urban-fl-nlp
    python setup_project.py mio_progetto       # crea ./mio_progetto
    python setup_project.py --here             # crea nella directory corrente
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent


# =============================================================================
# DIRECTORY DA CREARE
# =============================================================================

DIRECTORIES = [
    # configs
    "configs",
    "configs/data",
    "configs/strategies",
    "configs/experiments",

    # data (gitignored, ma le directory le creiamo lo stesso con .gitkeep)
    "data/raw/rdd2022",
    "data/raw/taco",
    "data/processed/rdd2022_clients",
    "data/synthetic",

    # src
    "src",
    "src/data",
    "src/models",
    "src/nlp",
    "src/federated",
    "src/federated/strategies",
    "src/evaluation",
    "src/utils",

    # scripts
    "scripts",

    # experiments (gitignored)
    "experiments/rq1",
    "experiments/rq2",
    "experiments/ablation",

    # notebooks
    "notebooks",

    # reports
    "reports/figures",
    "reports/tables",

    # tests
    "tests",
]


# Le directory che devono restare versionate anche se vuote ricevono un .gitkeep
GITKEEP_DIRS = [
    "data/raw/rdd2022",
    "data/raw/taco",
    "data/processed/rdd2022_clients",
    "data/synthetic",
    "experiments/rq1",
    "experiments/rq2",
    "experiments/ablation",
    "reports/figures",
    "reports/tables",
]


# Pacchetti Python che richiedono __init__.py
PYTHON_PACKAGES = [
    "src",
    "src/data",
    "src/models",
    "src/nlp",
    "src/federated",
    "src/federated/strategies",
    "src/evaluation",
    "src/utils",
    "tests",
]


# =============================================================================
# CONTENUTI DEI FILE
# =============================================================================

GITIGNORE = """\
# data e output sperimentali
data/raw/**
data/processed/**
data/synthetic/*.jsonl
experiments/**
!data/raw/.gitkeep
!data/raw/*/.gitkeep
!data/processed/.gitkeep
!data/processed/*/.gitkeep
!data/synthetic/.gitkeep
!experiments/.gitkeep
!experiments/*/.gitkeep

# checkpoint modelli
*.pt
*.pth
*.ckpt
*.onnx

# python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
build/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# virtual env
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# notebook
.ipynb_checkpoints/

# secrets
.env

# OS
.DS_Store
Thumbs.db

# mlflow / wandb
mlruns/
wandb/
"""


README = """\
# urban-fl-nlp

Monitoraggio urbano distribuito tramite Federated Learning + Natural Language Processing.

Implementa il meccanismo **NLP-Weighted Aggregation**: i pesi di aggregazione FL
dei client (veicoli) sono modulati dinamicamente da uno score di urgenza estratto
via NER + sentiment analysis dalle segnalazioni testuali dei cittadini.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparazione dati

```bash
# Scarica RDD2022 e crea lo split per paese (6 client FL Non-IID)
python scripts/prepare_rdd2022.py

# Genera il corpus sintetico di segnalazioni cittadine
python scripts/generate_synthetic_complaints.py
```

## Esperimenti

```bash
# RQ1 - baselines (FedAvg, FedProx, SCAFFOLD) vs Oracle vs NLP-Weighted
bash scripts/run_rq1.sh

# RQ2 - sweep su rumore NLP (100%, 90%, 80%, 70%, 50%)
bash scripts/run_rq2_sweep.sh

# Ablation - sensitivity analysis su cap e time decay
bash scripts/run_ablation.sh
```

## Struttura

- `configs/`        YAML config (niente parametri hardcoded nel codice)
- `src/`            codice sorgente (data, models, nlp, federated, evaluation)
- `scripts/`        entry point eseguibili
- `experiments/`    output delle run (gitignored)
- `notebooks/`      analisi e plotting
- `reports/`        figure, tabelle, report finale
- `tests/`          unit test

## Protocollo di validazione

Il protocollo a tre componenti (A + B + C) rompe la circolarita' della valutazione:

- **Componente A** - criticita' definita esogenamente in `src/evaluation/critical_classes.py`
- **Componente B** - braccio Oracle-weighted come upper bound (`src/federated/strategies/oracle_weighted.py`)
- **Componente C** - sweep controllato sul rumore NLP (`src/nlp/noise_injection.py`)
"""


REQUIREMENTS = """\
# core
torch>=2.1
torchvision>=0.16
numpy>=1.24
pandas>=2.0
pyyaml>=6.0

# federated learning
flwr>=1.7

# object detection
ultralytics>=8.0

# nlp
transformers>=4.35
spacy>=3.7
scikit-learn>=1.3

# config & experiment tracking
hydra-core>=1.3
mlflow>=2.8

# notebooks & viz
jupyter>=1.0
matplotlib>=3.7
seaborn>=0.13

# testing
pytest>=7.4
pytest-cov>=4.1
"""


PYPROJECT = """\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "urban-fl-nlp"
version = "0.1.0"
description = "Monitoraggio urbano distribuito via Federated Learning + NLP-Weighted Aggregation"
requires-python = ">=3.10"
readme = "README.md"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py310"
"""


ENV_EXAMPLE = """\
# copia in .env e compila i valori
MLFLOW_TRACKING_URI=./mlruns
WANDB_API_KEY=
HF_HOME=./.cache/huggingface
"""


# -------- config YAML stub --------

CONFIG_BASE = """\
# Iperparametri di default condivisi da tutti gli esperimenti.
# Le config specifiche in configs/experiments/ ereditano e sovrascrivono questi valori.

seed: 42
device: cuda

federated:
  num_rounds: 100
  clients_per_round: 6
  local_epochs: 3
  batch_size: 16

model:
  arch: yolov8n          # nano per partire, salire a yolov8s/m se serve
  num_classes: 4         # RDD2022: longitudinal, transverse, alligator, pothole
  img_size: 640

optimizer:
  name: sgd
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0005

logging:
  log_every_n_rounds: 1
  save_checkpoints: true
"""


CONFIG_DATA_RDD2022 = """\
# Split country-wise di RDD2022 -> 6 client FL Non-IID

dataset: rdd2022
root: data/processed/rdd2022_clients
clients:
  - japan
  - india
  - czech
  - norway
  - usa
  - china

# Filtro: solo positive images (almeno una bbox per immagine)
positive_only: true

# Classi e mapping (devono coincidere con src/evaluation/critical_classes.py)
classes:
  - longitudinal_crack    # D00
  - transverse_crack      # D10
  - alligator_crack       # D20
  - pothole               # D40
"""


CONFIG_FEDAVG = """\
# Baseline context-blind: FedAvg (McMahan et al., 2017)
defaults:
  - /base

strategy:
  name: fedavg
  # nessun parametro aggiuntivo: media pesata per |D_k|
"""


CONFIG_FEDPROX = """\
# Baseline regolarizzata: FedProx (Li et al., 2020)
defaults:
  - /base

strategy:
  name: fedprox
  mu: 0.01    # forza del termine prossimale
"""


CONFIG_SCAFFOLD = """\
# Baseline state-of-the-art per Non-IID: SCAFFOLD (Karimireddy et al., 2020)
defaults:
  - /base

strategy:
  name: scaffold
  # le control variate sono mantenute server-side e per-client
"""


CONFIG_ORACLE = """\
# Componente B del protocollo: upper bound informativo.
# Pesa i client usando direttamente la ground truth esogena di criticita'
# (definita in src/evaluation/critical_classes.py), SENZA passare dall'NLP.

defaults:
  - /base

strategy:
  name: oracle_weighted
  cap: 0.5            # cap sui pesi (anti oblio catastrofico)
  time_decay: 0.9     # costante di decadimento temporale per round
"""


CONFIG_NLP_WEIGHTED = """\
# Metodo proposto: NLP-Weighted Aggregation.
# Usa lo score di urgenza realistico (rumoroso) prodotto dal modulo NLP.

defaults:
  - /base

strategy:
  name: nlp_weighted
  cap: 0.5
  time_decay: 0.9

nlp:
  ner_model: dbmdz/bert-base-italian-xxl-cased   # placeholder
  sentiment_model: MilaNLProc/feel-it-italian-sentiment
  urgency_aggregator: mean                       # mean | max | weighted_recent
"""


CONFIG_RQ1 = """\
# RQ1 - confronto NLP-Weighted vs baselines context-blind + Oracle.
# Lancia tutte le strategie sugli stessi seed e split per confronto pareato.

experiment_name: rq1_baselines_vs_proposed

data:
  config: data/rdd2022_country_split

strategies:
  - strategies/fedavg
  - strategies/fedprox
  - strategies/scaffold
  - strategies/oracle_weighted
  - strategies/nlp_weighted

seeds: [42, 123, 2024, 7, 31]   # 5 seed per significativita' statistica

metrics:
  - global_map50
  - recall_at_critical    # recall sulla classe critica (Componente A)
  - rounds_to_target      # numero round per raggiungere recall target
"""


CONFIG_RQ2 = """\
# RQ2 - robustezza al rumore del segnale NLP (Componente C).
# Sweep sul livello di accuratezza simulata del classificatore di urgenza.

experiment_name: rq2_nlp_noise_sweep

data:
  config: data/rdd2022_country_split

base_strategy: strategies/nlp_weighted

noise_levels: [1.0, 0.9, 0.8, 0.7, 0.5]   # accuratezza simulata
seeds: [42, 123, 2024]

# Brace di confronto: la baseline FedAvg viene rilanciata per identificare
# la soglia di break-even (sotto la quale NLP-Weighted perde il vantaggio).
reference_baseline: strategies/fedavg

metrics:
  - global_map50
  - recall_at_critical
"""


CONFIG_ABLATION = """\
# Sensitivity analysis sugli iperparametri del meccanismo proposto.
# Caratterizza l'interazione cap x time_decay x local_epochs.

experiment_name: ablation_cap_decay

data:
  config: data/rdd2022_country_split

base_strategy: strategies/nlp_weighted

grid:
  cap: [0.2, 0.35, 0.5, 0.7, 1.0]
  time_decay: [0.7, 0.85, 0.9, 0.95, 1.0]
  local_epochs: [1, 3, 5]

seeds: [42, 123]
"""


# -------- stub Python --------

INIT_BARE = '"""urban-fl-nlp source package."""\n'

CRITICAL_CLASSES_STUB = '''\
"""
Componente A del protocollo di validazione.

Definisce ESOGENAMENTE quali classi sono critiche, prima di qualsiasi esperimento,
sulla base di letteratura tecnica e regolamentazioni di manutenzione urbana.
Una volta congelato, questo file NON va modificato per mantenere la validita'
del protocollo (rottura della circolarita').

Riferimenti:
- ASTM D6433: Standard Practice for Roads and Parking Lots Pavement Condition
  Index Surveys (PCI). Le potholes sono classificate come "high severity distress"
  con impatto diretto su sicurezza veicolare.
- TODO: aggiungere riferimento a regolamento comunale o linee guida MIT/ANAS.
"""

from __future__ import annotations

# Mapping nome_classe -> livello di criticita' (1 = bassa, 3 = alta)
CRITICALITY: dict[str, int] = {
    "longitudinal_crack": 1,
    "transverse_crack":   1,
    "alligator_crack":    2,
    "pothole":            3,   # critica per default
}

# Insieme delle classi considerate critiche per le metriche di valutazione
# (recall_at_critical e simili).
CRITICAL_CLASSES: frozenset[str] = frozenset(
    name for name, level in CRITICALITY.items() if level >= 3
)


def is_critical(class_name: str) -> bool:
    """True se class_name appartiene alle classi critiche definite esogenamente."""
    return class_name in CRITICAL_CLASSES
'''


STRATEGY_BASE_STUB = '''\
"""
Interfaccia comune per tutte le strategie di aggregazione FL.

Tutte le strategie (FedAvg, FedProx, SCAFFOLD, OracleWeighted, NLPWeighted)
ereditano da Strategy in modo che il meccanismo di pesatura sia l'UNICA
variabile fra i bracci sperimentali.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ClientUpdate:
    """Update inviato da un client al server in un round."""
    client_id: str
    num_examples: int
    parameters: object   # tensor / state_dict, dipende dall'implementazione
    metrics: dict


class Strategy(ABC):
    """Strategia di aggregazione federata."""

    name: str

    @abstractmethod
    def aggregate(
        self,
        round_idx: int,
        updates: list[ClientUpdate],
        context: dict | None = None,
    ) -> object:
        """Aggrega gli update dei client e ritorna i nuovi parametri globali.

        Args:
            round_idx: indice del round corrente.
            updates: lista degli update ricevuti dai client.
            context: informazioni out-of-band (es. urgency_score per NLPWeighted,
                     ground truth per OracleWeighted). None per le baseline
                     context-blind.
        """
        ...

    @abstractmethod
    def compute_weights(
        self,
        updates: list[ClientUpdate],
        context: dict | None = None,
    ) -> dict[str, float]:
        """Ritorna i pesi normalizzati {client_id: peso}."""
        ...
'''


NOISE_INJECTION_STUB = '''\
"""
Componente C del protocollo: iniezione di rumore controllato nel segnale NLP.

Permette di simulare classificatori di urgenza con accuratezza variabile
(100%, 90%, 80%, ...) per tracciare la curva di robustezza richiesta da RQ2.

Il modulo e' deliberatamente separato da urgency_score.py: la pipeline NLP
"vera" produce uno score, e il noise injector lo perturba. Cosi' possiamo
tenere fissa la pipeline e variare solo il rumore.
"""

from __future__ import annotations

import numpy as np


def inject_label_noise(
    urgency_scores: np.ndarray,
    ground_truth: np.ndarray,
    target_accuracy: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturba urgency_scores per simulare un classificatore con la accuratezza target.

    Args:
        urgency_scores: score di urgenza per zona/client, shape (N,).
        ground_truth: criticita' ground truth (Componente A), shape (N,).
        target_accuracy: accuratezza simulata in [0, 1].
        rng: generatore random per riproducibilita'.

    Returns:
        Score di urgenza perturbati, shape (N,).
    """
    raise NotImplementedError("TODO: implementare iniezione di rumore")
'''


PARTITIONER_STUB = '''\
"""
Partizionatore Non-IID per RDD2022.

Lo split country-wise produce 6 client FL con distribuzioni di classe
fortemente sbilanciate (Non-IID severo), come richiesto da RQ1.
"""

from __future__ import annotations

from pathlib import Path


def partition_rdd2022_by_country(
    raw_dir: Path,
    output_dir: Path,
    positive_only: bool = True,
) -> dict[str, int]:
    """Crea uno split country-wise di RDD2022.

    Args:
        raw_dir: directory contenente RDD2022 estratto.
        output_dir: directory di output (un sotto-folder per paese).
        positive_only: se True, mantiene solo immagini con almeno una bbox.

    Returns:
        Dizionario {country: num_immagini} con il count per client.
    """
    raise NotImplementedError("TODO: implementare partitioner")
'''


RUN_EXPERIMENT_STUB = '''\
#!/usr/bin/env python3
"""Entry point generico per lanciare un esperimento da config YAML."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path al config YAML dell'esperimento")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override del seed (per multi-run)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"),
                        help="Root delle directory di output")
    args = parser.parse_args()

    # TODO:
    # 1. carica config (con composizione hydra-style)
    # 2. crea output_dir/<exp_name>_seed<seed>/
    # 3. salva snapshot della config in quella directory
    # 4. setup logging + mlflow
    # 5. costruisci dati, modello, strategia, server FL
    # 6. esegui training
    # 7. salva metriche e checkpoint
    raise NotImplementedError


if __name__ == "__main__":
    main()
'''


# Mapping {path relativo: contenuto} per i file da creare.
FILES: dict[str, str] = {
    ".gitignore": GITIGNORE,
    "README.md": README,
    "requirements.txt": REQUIREMENTS,
    "pyproject.toml": PYPROJECT,
    ".env.example": ENV_EXAMPLE,

    "configs/base.yaml": CONFIG_BASE,
    "configs/data/rdd2022_country_split.yaml": CONFIG_DATA_RDD2022,
    "configs/strategies/fedavg.yaml": CONFIG_FEDAVG,
    "configs/strategies/fedprox.yaml": CONFIG_FEDPROX,
    "configs/strategies/scaffold.yaml": CONFIG_SCAFFOLD,
    "configs/strategies/oracle_weighted.yaml": CONFIG_ORACLE,
    "configs/strategies/nlp_weighted.yaml": CONFIG_NLP_WEIGHTED,
    "configs/experiments/rq1_baselines_vs_proposed.yaml": CONFIG_RQ1,
    "configs/experiments/rq2_nlp_noise_sweep.yaml": CONFIG_RQ2,
    "configs/experiments/ablation_cap_decay.yaml": CONFIG_ABLATION,

    "src/evaluation/critical_classes.py": CRITICAL_CLASSES_STUB,
    "src/federated/strategies/base.py": STRATEGY_BASE_STUB,
    "src/nlp/noise_injection.py": NOISE_INJECTION_STUB,
    "src/data/partitioner.py": PARTITIONER_STUB,

    "scripts/run_experiment.py": RUN_EXPERIMENT_STUB,
}


# =============================================================================
# LOGICA DI CREAZIONE
# =============================================================================

class Counter:
    def __init__(self) -> None:
        self.created = 0
        self.skipped = 0

    def mark_created(self) -> None:
        self.created += 1

    def mark_skipped(self) -> None:
        self.skipped += 1


def create_directory(path: Path, counter: Counter) -> None:
    if path.exists():
        counter.mark_skipped()
        return
    path.mkdir(parents=True, exist_ok=True)
    counter.mark_created()


def create_file(path: Path, content: str, counter: Counter) -> None:
    if path.exists():
        counter.mark_skipped()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    counter.mark_created()


def setup(root: Path) -> None:
    print(f"Setup del progetto in: {root.resolve()}")
    print("-" * 60)

    counter = Counter()

    # 1. directory
    print("Creazione directory...")
    for d in DIRECTORIES:
        create_directory(root / d, counter)

    # 2. .gitkeep nelle directory che restano vuote
    print("Aggiunta .gitkeep...")
    for d in GITKEEP_DIRS:
        create_file(root / d / ".gitkeep", "", counter)

    # 3. __init__.py per i pacchetti Python
    print("Creazione __init__.py...")
    for pkg in PYTHON_PACKAGES:
        create_file(root / pkg / "__init__.py", INIT_BARE, counter)

    # 4. file meta e stub
    print("Creazione file meta e stub...")
    for relpath, content in FILES.items():
        create_file(root / relpath, content, counter)

    print("-" * 60)
    print(f"Creati: {counter.created}    Saltati (gia' esistenti): {counter.skipped}")
    print("\nProssimi passi:")
    print(f"  cd {root}")
    print("  python -m venv .venv && source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  git init && git add . && git commit -m 'initial structure'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea la struttura del progetto urban-fl-nlp."
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="urban-fl-nlp",
        help="Nome della directory di progetto (default: urban-fl-nlp)",
    )
    parser.add_argument(
        "--here",
        action="store_true",
        help="Crea la struttura nella directory corrente invece di una sottocartella",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd() if args.here else Path.cwd() / args.name
    setup(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())