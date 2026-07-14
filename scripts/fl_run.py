"""
FL run generico: FedAvg vanilla vs Oracle-weighted, su 2 o 6 client RDD2022.

Evoluzione di scripts/fl_minimal.py. Aggiunge:
    - selezione della strategia via --strategy {fedavg, oracle},
    - lettura dei pesi oracle dal global_manifest.json,
    - parametri --alpha e --cap della Componente B del protocollo,
    - 6 client di default (i 6 paesi del paper Saha 2024).

Uso:
    # FedAvg vanilla, 6 client, run rapida
    python scripts/fl_run.py --strategy fedavg --rounds 3 --local-epochs 1 --imgsz 416

    # Oracle-weighted, alpha=0.5, cap=0.5 (default), 6 client
    python scripts/fl_run.py --strategy oracle --rounds 3 --local-epochs 1 --imgsz 416

    # Oracle puro (alpha=0.0): peso solo dalla criticita'
    python scripts/fl_run.py --strategy oracle --alpha 0.0 --rounds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import flwr as fl
from flwr.common import Context, NDArrays, Scalar



PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENTS_ROOT = PROJECT_ROOT / "data" / "processed" / "rdd2022_clients"
RUNS_ROOT = PROJECT_ROOT / "experiments" / "fl_runs"
MANIFEST_PATH = CLIENTS_ROOT / "global_manifest.json"

# Aggiungi src/ al path per importare la nostra strategia
sys.path.insert(0, str(PROJECT_ROOT))
from src.federated.strategies.oracle_weighted import (  # noqa: E402
    OracleWeightedFedAvg,
    load_oracle_weights,
)
from src.federated.strategies.nlp_weighted import (  # noqa: E402
    NLPWeightedFedAvg,
    load_oracle_urgency,
    load_noisy_oracle_urgency,
)


# Lista canonica dei client (ordine = partition_id 0..N-1)
ALL_CLIENT_NAMES = [
    "czech", "india", "japan", "norway", "united_states", "china_motorbike",
]
MINIMAL_CLIENT_NAMES = ["japan", "india"]   # per debug rapido

# Classe critica esogena (Componente A)
CRITICAL_CLASS = "D40"


# ---------------------------------------------------------------------------
# UTILS: SERIALIZZAZIONE PESI (identica a fl_minimal.py)
# ---------------------------------------------------------------------------

def get_parameters_from_model(model: torch.nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_on_model(model: torch.nn.Module, parameters: NDArrays) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"Numero di parametri mismatch: modello ha {len(keys)} tensori, "
            f"Flower ha mandato {len(parameters)}."
        )
    state_dict = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(keys, parameters)
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# CLIENT FLOWER (identico a fl_minimal.py, riportato per autonomia del file)
# ---------------------------------------------------------------------------

class YoloFlowerClient(fl.client.NumPyClient):

    def __init__(
            self,
            client_name: str,
            partition_id: int,
            data_yaml: Path,
            local_epochs: int,
            batch: int,
            imgsz: int,
            device: str,
            run_root: Path,
            seed: int = 42,
    ) -> None:
        self.client_name = client_name
        self.partition_id = partition_id
        self.data_yaml = data_yaml
        self.local_epochs = local_epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.run_root = run_root
        self.seed = seed

        self.num_train = sum(
            1 for _ in (data_yaml.parent / "train" / "images").iterdir()
        )
        self._round = 0

    def _build_fresh_model(self):
        """Crea un YOLO con la testa a 4 classi (RDD2022 standard)."""
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        model = YOLO("yolov8n.pt")
        model.model.nc = 4
        model.model.names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        new_model = DetectionModel(cfg=model.model.yaml, nc=4, verbose=False)
        pretrained_state = model.model.state_dict()
        new_state = new_model.state_dict()
        for k in new_state:
            if k in pretrained_state and pretrained_state[k].shape == new_state[k].shape:
                new_state[k] = pretrained_state[k]
        new_model.load_state_dict(new_state)
        model.model = new_model
        return model

    def _build_model_with_params(self, parameters: NDArrays):
        model = self._build_fresh_model()
        set_parameters_on_model(model.model, parameters)
        return model

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        model = self._build_fresh_model()
        return get_parameters_from_model(model.model)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self._round += 1
        print(f"\n[{self.client_name}] FIT round {self._round} "
              f"(epochs={self.local_epochs}, n_train={self.num_train})")

        model = self._build_model_with_params(parameters)
        model.train(
            data=str(self.data_yaml),
            epochs=self.local_epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            device=self.device,
            workers=2,
            project=str(self.run_root / "clients"),
            name=f"{self.client_name}_r{self._round}",
            exist_ok=True,
            patience=self.local_epochs + 1,
            plots=False,
            verbose=False,
            seed=self.seed,
        )

        new_parameters = get_parameters_from_model(model.model)
        return new_parameters, self.num_train, {"partition_id": self.partition_id}

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        print(f"[{self.client_name}] EVALUATE round {self._round}")
        model = self._build_model_with_params(parameters)
        metrics = model.val(
            data=str(self.data_yaml),
            device=self.device,
            project=str(self.run_root / "evals"),
            name=f"{self.client_name}_r{self._round}",
            exist_ok=True,
            verbose=False,
            plots=False,
        )

        map50 = float(metrics.box.map50)
        map_5095 = float(metrics.box.map)

        # Recall per-class: estraiamo quella della classe critica (D40, idx=3)
        # Utile per la metrica recall_at_critical della RQ1
        try:
            # AP@50 per classe (soglia-indipendente, stabile fra round)
            class_ap50 = metrics.box.ap50
            critical_ap50 = float(class_ap50[3]) if len(class_ap50) > 3 else 0.0
        except (AttributeError, IndexError):
            critical_ap50 = 0.0

        return (
            1.0 - map50,
            self.num_train,
            {
                "map50": map50,
                "map50_95": map_5095,
                "critical_ap50": critical_ap50,  # <-- rinominata
                "partition_id": self.partition_id,
            },
        )


# ---------------------------------------------------------------------------
# CLIENT_FN
# ---------------------------------------------------------------------------

def make_client_fn(args: argparse.Namespace, run_root: Path,
                    client_names: list[str]):

    def client_fn(context: Context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])
        client_name = client_names[partition_id]

        data_yaml = CLIENTS_ROOT / client_name / "data.yaml"
        if not data_yaml.exists():
            sys.exit(f"ERRORE: data.yaml non trovato: {data_yaml}")

        return YoloFlowerClient(
            client_name=client_name,
            data_yaml=data_yaml,
            partition_id=partition_id,
            local_epochs=args.local_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            run_root=run_root,
        ).to_client()

    return client_fn


# ---------------------------------------------------------------------------
# AGGREGATORI DI METRICHE
# ---------------------------------------------------------------------------

def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Aggrega le metriche dei client pesando per #esempi.

    Esclude 'partition_id' che e' solo un metadato di routing per il server,
    non una metrica vera.
    """
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    aggregated: dict[str, float] = {}
    for key in metrics[0][1].keys():
        if key == "partition_id":
            continue
        aggregated[key] = sum(n * m[key] for n, m in metrics) / total
    return aggregated


# ---------------------------------------------------------------------------
# SELEZIONE DELLA STRATEGIA
# ---------------------------------------------------------------------------

def build_strategy(args: argparse.Namespace,
                    client_names: list[str]) -> fl.server.strategy.Strategy:
    """Costruisce la Strategy in base a --strategy."""
    n = len(client_names)
    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n,
        min_evaluate_clients=n,
        min_available_clients=n,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    if args.strategy == "fedavg":
        return fl.server.strategy.FedAvg(**common_kwargs)

    if args.strategy == "oracle":
        oracle_w = load_oracle_weights(
            MANIFEST_PATH,
            critical_class=CRITICAL_CLASS,
            client_names=client_names,
        )
        return OracleWeightedFedAvg(
            oracle_weights=oracle_w,
            client_names=client_names,
            alpha=args.alpha,
            cap=args.cap,
            **common_kwargs,
        )
    if args.strategy == "nlp":
        if args.urgency_provider == "oracle":
            provider = load_oracle_urgency(
                MANIFEST_PATH,
                critical_class=CRITICAL_CLASS,
                client_names=client_names,
            )
        elif args.urgency_provider == "noisy_oracle":
            provider = load_noisy_oracle_urgency(
                MANIFEST_PATH,
                critical_class=CRITICAL_CLASS,
                client_names=client_names,
                accuracy=args.noise_accuracy,
                seed=args.noise_seed if args.noise_seed is not None else args.seed,
            )
        elif args.urgency_provider == "nlp_pipeline":
            from src.nlp.urgency_provider import NLPUrgencyProvider
            provider = NLPUrgencyProvider(
                complaints_csv_path="data/nlp/synthetic_complaints_realistic.csv",
                all_clients=client_names,
                verbose=True,
            )
        else:
            raise ValueError(
                f"urgency_provider sconosciuto: {args.urgency_provider}"
            )
        return NLPWeightedFedAvg(
            urgency_provider=provider,
            client_names=client_names,
            alpha=args.alpha,
            cap=args.cap,
            time_decay=args.time_decay,
            **common_kwargs,
        )

    raise ValueError(f"Strategia sconosciuta: {args.strategy}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", choices=["fedavg", "oracle", "nlp"], default="fedavg",
        help="Quale strategy usare per l'aggregazione",
    )
    parser.add_argument(
        "--clients", choices=["minimal", "all"], default="all",
        help="'minimal' = japan + india; 'all' = 6 client del paper Saha 2024",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", type=str, default="0")

    # Iperparametri specifici di Oracle (e in futuro NLP-Weighted)
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Mix tra peso FedAvg e peso oracle (1.0 = pure FedAvg, "
             "0.0 = pure oracle). Default: 0.5",
    )
    parser.add_argument(
        "--cap", type=float, default=0.5,
        help="Tetto massimo per il peso di un singolo client. Default: 0.5",
    )
    parser.add_argument(
        "--time-decay", type=float, default=1.0,
        help="Time decay del boost di urgenza per round (in (0, 1]). "
             "1.0 = nessun decay. Default: 1.0",
    )
    parser.add_argument(
        "--urgency-provider", choices=["oracle", "noisy_oracle", "nlp_pipeline"], default="oracle",
        help="Sorgente dell'urgency score. 'oracle' = pesi statici dal manifest. "
             "'noisy_oracle' = oracle + rumore (per RQ2). Default: oracle",
    )
    parser.add_argument(
        "--noise-accuracy", type=float, default=1.0,
        help="Accuratezza simulata del classificatore NLP in [0, 1]. "
             "Usato solo se --urgency-provider noisy_oracle. Default: 1.0",
    )
    parser.add_argument(
        "--noise-seed", type=int, default=None,
        help="Seed per la sequenza di permutazioni del noisy_oracle. Default: 42",
    )
    parser.add_argument(
    "--seed", type=int, default=42,
    help="Seed globale per Ultralytics (training/val) e per il numpy "
         "interno. Default: 42",
    )


    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client_names = (ALL_CLIENT_NAMES if args.clients == "all"
                                     else MINIMAL_CLIENT_NAMES)

    # Verifica che tutti i data.yaml esistano
    for name in client_names:
        yaml_path = CLIENTS_ROOT / name / "data.yaml"
        if not yaml_path.exists():
            sys.exit(f"ERRORE: {yaml_path} non trovato.")

    # Output run-specifico (incluso il nome strategia per disambiguare)
    run_id = f"{args.strategy}_{args.clients}_{int(time.time())}"
    run_root = RUNS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FL run: {args.strategy} su {len(client_names)} client")
    print(f"  client       : {client_names}")
    print(f"  rounds       : {args.rounds}")
    print(f"  local_epochs : {args.local_epochs}")
    print(f"  batch        : {args.batch}, imgsz: {args.imgsz}")
    print(f"  device       : {args.device}")
    if args.strategy == "oracle":
        print(f"  alpha        : {args.alpha}")
        print(f"  cap          : {args.cap}")
    if args.strategy == "nlp":
        print(f"  provider     : {args.urgency_provider}")
        print(f"  alpha        : {args.alpha}")
        print(f"  cap          : {args.cap}")
        print(f"  time_decay   : {args.time_decay}")
        if args.urgency_provider == "noisy_oracle":
            print(f"  noise_accuracy: {args.noise_accuracy}")
            print(f"  noise_seed   : {args.noise_seed}")
    print(f"  output       : {run_root}")
    print("=" * 70)

    strategy = build_strategy(args, client_names)

    #client_resources = {
    #    "num_cpus": 2,
    #    "num_gpus": 1.0 if args.device != "cpu" else 0.0,
    #}

    client_resources = {
        "num_cpus": 2,
        "num_gpus": 0.16 if args.device != "cpu" else 0.0,
    }

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(args, run_root, client_names),
        num_clients=len(client_names),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    # Risultati
    print("\n" + "=" * 70)
    print(f"RISULTATI FINALI [{args.strategy}]")
    print("=" * 70)
    print("\nMetriche aggregate per round:")
    if history.metrics_distributed:
        for key, values in history.metrics_distributed.items():
            print(f"\n  {key}:")
            for round_idx, value in values:
                print(f"    round {round_idx}: {value:.4f}")
    else:
        print("  (nessuna metrica registrata)")

    # Salva i risultati in JSON per confronto fra strategie
    import json
    results_path = run_root / "results.json"
    results_path.write_text(json.dumps({
        "strategy": args.strategy,
        "clients": client_names,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "alpha": args.alpha if args.strategy == "oracle" else None,
        "cap": args.cap if args.strategy == "oracle" else None,
        "time_decay": args.time_decay if args.strategy == "nlp" else None,
        "urgency_provider": args.urgency_provider if args.strategy == "nlp" else None,
        "noise_accuracy": args.noise_accuracy if args.urgency_provider == "noisy_oracle" else None,
        "noise_seed": args.noise_seed if args.urgency_provider == "noisy_oracle" else None,
        "seed": args.seed,
        "metrics_distributed": {
            k: [[r, v] for r, v in vals]
            for k, vals in history.metrics_distributed.items()
        },
        "losses_distributed": [[r, v] for r, v in history.losses_distributed]
                              if history.losses_distributed else [],
    }, indent=2), encoding="utf-8")
    print(f"\nResults JSON: {results_path}")
    print(f"Artefatti in: {run_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())