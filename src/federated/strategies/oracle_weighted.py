"""
Oracle-weighted FedAvg: braccio "upper bound informativo" del protocollo.

Implementa la Componente B della validazione (Sezione 4.2 del report):
pesa i client usando direttamente la ground truth ESOGENA di criticita'
(Componente A), senza passare dalla pipeline NLP. Definisce quindi quanto
beneficio si potrebbe ottenere "in principio" se il segnale di urgenza
fosse perfetto.

L'aggregazione avviene secondo:

    weight_k(round=t) = alpha * (n_k / Sum n_j) + (1 - alpha) * crit_k

dove:
    n_k       = numero di esempi del client k (peso FedAvg classico)
    crit_k    = peso di criticita' del client k (normalizzato a sommare a 1),
                derivato dalla concentrazione della classe critica
    alpha     = mix in [0, 1]:
                    1.0  -> FedAvg vanilla (no criticity)
                    0.0  -> pesatura puramente per criticita'
                    0.5  -> mix bilanciato (default)

Inoltre applica il CAP sui pesi (Sezione 5 del report): nessun client puo'
ricevere piu' di `cap` (es. 0.5) del peso totale. L'eccesso viene redistribuito
proporzionalmente sugli altri client. Questo previene l'oblio catastrofico.

Per ora i pesi di criticita' sono STATICI: derivano dalla quota di bbox della
classe critica nel dataset di ciascun client, calcolata una volta sola dal
global_manifest.json. La versione DINAMICA (peso oracle che varia di round
in round per simulare picchi di lamentele) sara' un secondo step.
"""

from __future__ import annotations

import json
from logging import WARNING
from pathlib import Path
from typing import Optional

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


# ---------------------------------------------------------------------------
# COSTRUZIONE DEI PESI DI CRITICITA' DAI MANIFEST
# ---------------------------------------------------------------------------

def load_oracle_weights(
    manifest_path: Path,
    critical_class: str,
    client_names: list[str],
) -> dict[str, float]:
    """Calcola i pesi di criticita' (normalizzati) per ciascun client.

    Per ogni client, il peso e' la quota di bbox della classe critica nel
    suo dataset, normalizzata sulla somma dei totali di tutti i client.

    Args:
        manifest_path: path a global_manifest.json prodotto da
            convert_voc_to_yolo_all.py.
        critical_class: nome della classe critica (es. "D40" per pothole).
        client_names: lista dei nomi dei client (lowercase) nell'ordine
            atteso da Flower (idx 0..N-1).

    Returns:
        Dict {client_name: weight} con i pesi normalizzati a sommare a 1.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Estrai bbox totali della classe critica per ogni client conosciuto
    crit_counts: dict[str, int] = {}
    for stats in manifest["per_country_stats"]:
        client = stats["client_name"]   # gia' lowercase nel manifest
        train_count = stats["train"]["boxes_per_class"].get(critical_class, 0)
        valid_count = stats["valid"]["boxes_per_class"].get(critical_class, 0)
        crit_counts[client] = train_count + valid_count

    # Filtra ai client effettivamente in uso e verifica copertura
    missing = [c for c in client_names if c not in crit_counts]
    if missing:
        raise ValueError(
            f"Client non trovati nel manifest: {missing}. "
            f"Manifest contiene: {list(crit_counts.keys())}"
        )

    selected = {c: crit_counts[c] for c in client_names}
    total = sum(selected.values())
    if total == 0:
        raise ValueError(
            f"Nessuna bbox della classe critica '{critical_class}' "
            f"trovata nei client {client_names}. Verifica che la classe "
            f"sia spelled correctly."
        )

    return {c: count / total for c, count in selected.items()}


# ---------------------------------------------------------------------------
# STRATEGIA
# ---------------------------------------------------------------------------

class OracleWeightedFedAvg(FedAvg):
    """FedAvg con pesi di aggregazione informati dalla ground truth esogena.

    Eredita tutto da FedAvg (configure_fit, evaluate, ecc.) e fa override
    SOLO di aggregate_fit per applicare i pesi oracle + cap.
    """

    def __init__(
        self,
        oracle_weights: dict[str, float],
        client_names: list[str],
        alpha: float = 0.5,
        cap: float = 0.5,
        **kwargs,
    ) -> None:
        """
        Args:
            oracle_weights: dict {client_name: criticity_weight}, sommano a 1.
            client_names: ordine canonico dei client (idx 0..N-1).
            alpha: mix tra peso FedAvg e peso oracle (in [0, 1]).
            cap: tetto massimo per il peso di un singolo client (in (0, 1]).
            kwargs: passati a FedAvg.__init__ (fraction_fit, min_*_clients...).
        """
        super().__init__(**kwargs)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha deve essere in [0, 1], dato: {alpha}")
        if not (0.0 < cap <= 1.0):
            raise ValueError(f"cap deve essere in (0, 1], dato: {cap}")
        if abs(sum(oracle_weights.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"oracle_weights deve sommare a 1, somma: "
                f"{sum(oracle_weights.values())}"
            )

        self.oracle_weights = oracle_weights
        self.client_names = client_names
        self.alpha = alpha
        self.cap = cap

        # Diagnostica all'init: stampa i pesi per il log dell'esperimento
        log(
            WARNING,
            "OracleWeightedFedAvg init: alpha=%.2f, cap=%.2f",
            alpha, cap,
        )
        log(WARNING, "Oracle weights (criticita'):")
        for name, w in oracle_weights.items():
            log(WARNING, "  %s: %.4f", name, w)

    # -- mapping client_proxy.cid -> client_name ------------------------

    
    # -- calcolo dei pesi di aggregazione --------------------------------

    def _compute_aggregation_weights(
        self,
        client_results: list[tuple[ClientProxy, FitRes]],
    ) -> list[float]:
        """Calcola un peso per ciascun client del round corrente.

        Combina:
            - peso FedAvg basato su num_examples (n_k / sum n_j),
            - peso oracle basato su criticita' (gia' normalizzato),
            - cap sul singolo client,
            - rinormalizzazione finale.
        """
        # 1. Estrai n_k e oracle_w_k per ciascun client del round
        n_k = np.array([fit_res.num_examples for _, fit_res in client_results],
                        dtype=float)
        n_total = n_k.sum()
        if n_total == 0:
            log(WARNING, "Tutti i client hanno num_examples=0, fallback su pesi uniformi")
            return [1.0 / len(client_results)] * len(client_results)
        fedavg_w = n_k / n_total

        oracle_w = np.array([
            self.oracle_weights[
                self.client_names[int(fit_res.metrics["partition_id"])]
            ]
            for _, fit_res in client_results
        ], dtype=float)
        # Rinormalizza oracle_w sul subset partecipante a questo round
        # (importante quando fraction_fit < 1.0 e non tutti i client partecipano)
        oracle_w = oracle_w / oracle_w.sum() if oracle_w.sum() > 0 else fedavg_w

        # 2. Mix
        weights = self.alpha * fedavg_w + (1 - self.alpha) * oracle_w

        # 3. Applica il cap iterativamente
        weights = self._apply_cap(weights)

        # 4. Sanity check: somma a 1
        s = weights.sum()
        if abs(s - 1.0) > 1e-6:
            weights = weights / s

        return weights.tolist()

    def _apply_cap(self, weights: np.ndarray) -> np.ndarray:
        """Applica il tetto massimo iterativamente.

        Quando un client supera il cap, lo abbassa al cap e redistribuisce
        l'eccesso sugli altri (proporzionalmente ai loro pesi correnti).
        Iterativo perche' la redistribuzione puo' a sua volta far sforare
        un altro client se i pesi sono molto sbilanciati.
        """
        weights = weights.copy()
        max_iter = 100
        for _ in range(max_iter):
            over = weights > self.cap
            if not over.any():
                return weights

            # Quanto eccesso totale c'e' da redistribuire
            excess = (weights[over] - self.cap).sum()
            weights[over] = self.cap

            # Redistribuiamo sui client sotto il cap, proporzionalmente
            under = ~over
            if not under.any():
                # Tutti sopra il cap (impossibile se cap*N >= 1)
                log(WARNING, "Nessun client sotto cap: %s", weights)
                break
            under_weights = weights[under]
            under_total = under_weights.sum()
            if under_total == 0:
                # Distribuzione uniforme di emergenza
                weights[under] = excess / under.sum()
            else:
                weights[under] += excess * (under_weights / under_total)

        return weights

    # -- override aggregate_fit -------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggrega i pesi dei client usando weights = mix(fedavg, oracle).

        Replica la struttura di FedAvg.aggregate_fit ma con i nostri pesi.
        """
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Calcola i pesi del round corrente
        weights = self._compute_aggregation_weights(results)

        # Logga i pesi per il debug
        log(WARNING, "Round %d aggregation weights:", server_round)
        for (_, fit_res), w in zip(results, weights):
            name = self.client_names[int(fit_res.metrics["partition_id"])]
            log(WARNING, "  %s: %.4f", name, w)

        # Costruiamo (parameters, num_examples=peso) per riusare aggregate()
        # di Flower. aggregate() fa media pesata, e il peso e' num_examples;
        # quindi convertiamo i nostri weights in "num_examples sintetici"
        # mantenendo le proporzioni. Moltiplichiamo per un fattore grande
        # per evitare arrotondamenti a zero.
        SCALE = 10_000
        weighted_results: list[tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), int(w * SCALE))
            for (_, fit_res), w in zip(results, weights)
        ]

        aggregated_ndarrays = aggregate(weighted_results)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggrega le metriche di fit (se i client le ritornano)
        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics)
                            for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return aggregated_parameters, metrics_aggregated