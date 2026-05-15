"""
Federated Learning minimo: FedAvg + YOLOv8n su 2 client RDD2022 (Japan + India).

Obiettivo Milestone 2: dimostrare end-to-end che il bridge YOLO <-> Flower
funziona e che la mAP@50 globale cresce di round in round con FedAvg vanilla.

Volutamente NON ottimizzato:
    - 2 client invece di 6 (scaleremo dopo).
    - Strategia FedAvg built-in di Flower (non scriviamo ancora la nostra).
    - Ogni round istanzia un nuovo modello YOLO (sub-ottimale ma chiaro).
    - Niente Hydra, niente MLflow, niente test custom.

Una volta che questo funziona, lo evolveremo in una versione produzione
con tutti i 6 client e le strategie custom (Oracle, NLP-Weighted).

Uso:
    python scripts/fl_minimal.py
    python scripts/fl_minimal.py --rounds 3 --local-epochs 5
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import flwr as fl
from flwr.common import Context, NDArrays, Scalar


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENTS_ROOT = PROJECT_ROOT / "data" / "processed" / "rdd2022_clients"
RUNS_ROOT = PROJECT_ROOT / "experiments" / "fl_minimal"

# I 2 client per la run minimale: scelti perche' hanno distribuzioni di classe
# molto diverse fra loro (Japan e' alligator-heavy, India e' pothole-heavy).
# Vedere reports/figures/class_distribution_heatmap.png.
CLIENT_NAMES = ["japan", "india"]


# ---------------------------------------------------------------------------
# UTILS: SERIALIZZAZIONE PESI PYTORCH <-> NUMPY (formato Flower)
# ---------------------------------------------------------------------------

def get_parameters_from_model(model: torch.nn.Module) -> NDArrays:
    """state_dict() -> lista di NumPy arrays (formato che Flower trasmette).

    Includiamo TUTTI i tensori (anche buffer come BatchNorm running_mean): in
    YOLO sono importanti per la coerenza del modello globale.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_on_model(model: torch.nn.Module, parameters: NDArrays) -> None:
    """Lista di NumPy arrays -> state_dict() applicato in-place sul modello.

    Deve mantenere la stessa cardinalita' e ordine di get_parameters_from_model.
    """
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
# CLIENT FLOWER
# ---------------------------------------------------------------------------

class YoloFlowerClient(fl.client.NumPyClient):
    """Un client = un paese.

    Ogni round, fit():
        1. riceve i pesi globali dal server (NDArrays),
        2. li carica in un YOLO locale,
        3. addestra per local_epochs sui dati del paese,
        4. ritorna i nuovi pesi + dimensione del dataset (per la media pesata).

    evaluate():
        1. riceve i pesi globali,
        2. valuta su valid/ del paese,
        3. ritorna mAP@50 e mAP@50-95 come metriche.
    """

    def __init__(
        self,
        client_name: str,
        data_yaml: Path,
        local_epochs: int,
        batch: int,
        imgsz: int,
        device: str,
        run_root: Path,
    ) -> None:
        self.client_name = client_name
        self.data_yaml = data_yaml
        self.local_epochs = local_epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.run_root = run_root

        # Numero di immagini di training -> peso FedAvg
        self.num_train = sum(
            1 for _ in (data_yaml.parent / "train" / "images").iterdir()
        )

        # Round counter (per nominare le sottocartelle Ultralytics)
        self._round = 0

    # -- helper: costruisce un modello YOLO con pesi correnti -------------

    def _build_model_with_params(self, parameters: NDArrays):
        """Istanzia YOLOv8n con la testa giusta (4 classi) e applica i pesi."""
        model = self._build_fresh_model()
        set_parameters_on_model(model.model, parameters)
        return model

    def _build_fresh_model(self):
        """Crea un YOLO con la testa a 4 classi come da nostro data.yaml.

        Trick: facciamo un brevissimo train(epochs=0) per forzare Ultralytics
        a riallineare la testa di detection su nc=4. Niente training reale,
        solo la riconfigurazione della testa.
        """
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # Riconfiguriamo la testa: il modo affidabile e' lasciare che Ultralytics
        # lo faccia leggendo il nostro data.yaml.
        model.model.nc = 4
        model.model.names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        # Forziamo la ricostruzione della testa con nc=4
        from ultralytics.nn.tasks import DetectionModel
        new_model = DetectionModel(cfg=model.model.yaml, nc=4, verbose=False)
        # Trasferiamo i pesi del backbone (tutto tranne la testa con shape diversa)
        pretrained_state = model.model.state_dict()
        new_state = new_model.state_dict()
        for k in new_state:
            if k in pretrained_state and pretrained_state[k].shape == new_state[k].shape:
                new_state[k] = pretrained_state[k]
        new_model.load_state_dict(new_state)
        model.model = new_model
        return model

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Ritorna i pesi iniziali del modello con testa a 4 classi."""
        model = self._build_fresh_model()
        return get_parameters_from_model(model.model)

    # -- fit: training locale ---------------------------------------------

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self._round += 1
        print(f"\n[{self.client_name}] FIT round {self._round} "
              f"(epochs={self.local_epochs}, n_train={self.num_train})")

        model = self._build_model_with_params(parameters)

        # Ultralytics gestisce il proprio training loop. Lo invochiamo
        # con epochs=local_epochs e i pesi di partenza gia' caricati.
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
            patience=self.local_epochs + 1,  # mai early stop
            plots=False,
            verbose=False,
            seed=42,
        )

        # Estrai pesi aggiornati
        new_parameters = get_parameters_from_model(model.model)
        return new_parameters, self.num_train, {}

    # -- evaluate: validazione locale -------------------------------------

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

        # Flower vuole una "loss" come primo valore. Per detection non c'e' una
        # loss naturale post-training; usiamo (1 - mAP@50) come proxy: piu' e'
        # bassa, meglio e'.
        map50 = float(metrics.box.map50)
        map = float(metrics.box.map)

        return (
            1.0 - map50,                              # "loss"
            self.num_train,                           # peso del client
            {"map50": map50, "map50_95": map},        # metriche
        )


# ---------------------------------------------------------------------------
# CLIENT_FN: factory invocata da Flower per ciascun client del round
# ---------------------------------------------------------------------------

def make_client_fn(args: argparse.Namespace, run_root: Path):
    """Costruisce la closure che Flower chiamera' per ottenere ogni client."""

    def client_fn(context: Context) -> fl.client.Client:
        # In simulation, context.node_config contiene un id intero (0..N-1).
        # Lo mappiamo sul nome del paese.
        partition_id = int(context.node_config["partition-id"])
        client_name = CLIENT_NAMES[partition_id]

        data_yaml = CLIENTS_ROOT / client_name / "data.yaml"
        if not data_yaml.exists():
            sys.exit(f"ERRORE: data.yaml non trovato per {client_name}: "
                     f"{data_yaml}")

        return YoloFlowerClient(
            client_name=client_name,
            data_yaml=data_yaml,
            local_epochs=args.local_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            run_root=run_root,
        ).to_client()

    return client_fn


# ---------------------------------------------------------------------------
# AGGREGATORI DI METRICHE (server-side)
# ---------------------------------------------------------------------------

def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Aggrega le metriche dei client pesando per #esempi.

    Flower passa una lista di (num_examples, metrics_dict) e si aspetta
    indietro un dict aggregato.
    """
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    aggregated: dict[str, float] = {}
    for key in metrics[0][1].keys():
        weighted_sum = sum(n * m[key] for n, m in metrics)
        aggregated[key] = weighted_sum / total_examples
    return aggregated


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3,
                        help="Numero di round federati")
    parser.add_argument("--local-epochs", type=int, default=5,
                        help="Epoch di training locali per round")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0",
                        help="Device GPU ('0') o 'cpu'")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Verifica dataset
    for name in CLIENT_NAMES:
        yaml_path = CLIENTS_ROOT / name / "data.yaml"
        if not yaml_path.exists():
            sys.exit(f"ERRORE: {yaml_path} non trovato. "
                     f"Hai eseguito convert_voc_to_yolo_all.py?")

    import time
    run_id = f"run_{int(time.time())}"
    run_root = RUNS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FL MINIMAL - FedAvg vanilla con YOLOv8n su {len(CLIENT_NAMES)} client")
    print(f"  client       : {CLIENT_NAMES}")
    print(f"  rounds       : {args.rounds}")
    print(f"  local_epochs : {args.local_epochs}")
    print(f"  batch        : {args.batch}")
    print(f"  device       : {args.device}")
    print(f"  output       : {run_root}")
    print("=" * 70)

    # Strategia: FedAvg built-in con aggregatore di metriche custom
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                    # tutti i client partecipano
        fraction_evaluate=1.0,
        min_fit_clients=len(CLIENT_NAMES),
        min_evaluate_clients=len(CLIENT_NAMES),
        min_available_clients=len(CLIENT_NAMES),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # GPU sharing: in simulation tutti i client girano nello stesso processo.
    # client_resources dice a Ray quante "unita'" di GPU servono. Mettendo 1.0
    # i client girano in serie (uno alla volta). Su 4070 8GB con YOLOv8n e'
    # la scelta sicura.
    client_resources = {"num_cpus": 2, "num_gpus": 1.0 if args.device != "cpu" else 0.0}

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(args, run_root),
        num_clients=len(CLIENT_NAMES),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    # Stampa storia
    print("\n" + "=" * 70)
    print("RISULTATI FINALI")
    print("=" * 70)
    print("\nMetriche distribuite (mAP@50 globale aggregato per round):")
    if history.metrics_distributed:
        for key, values in history.metrics_distributed.items():
            print(f"\n  {key}:")
            for round_idx, value in values:
                print(f"    round {round_idx}: {value:.4f}")
    else:
        print("  (nessuna metrica registrata)")

    print(f"\nArtefatti in: {run_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())