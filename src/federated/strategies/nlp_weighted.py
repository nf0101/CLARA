"""
NLP-Weighted FedAvg: metodo proposto del progetto.

La Strategy pesa i client usando uno score di URGENZA per client per round,
prodotto out-of-band da un UrgencyProvider. La pipeline NLP "vera" (NER +
sentiment + corpus testuale) sara' un provider concreto fra gli altri.

Architettura:

    +---------------------+      get_urgency(round, client_names)
    | NLPWeightedFedAvg   | -----------------------------------+
    |   - cap, time decay |                                    |
    |   - aggregation     |                                    v
    +---------------------+                       +-------------------------+
                                                   | UrgencyProvider (ABC)   |
                                                   +-------------------------+
                                                              ^
                                                              |
                            +---------------------------+-----+-----+--------------------+
                            |                           |           |                    |
                  OracleUrgencyProvider  NoisyOracleProvider  TimeVarying...     RealNLPProvider
                  (Step 1)                (Step 2 - RQ2)     (Step 3)            (M4)

Step 1: solo OracleUrgencyProvider. Quando lo si usa, NLPWeightedFedAvg deve
produrre risultati IDENTICI a OracleWeightedFedAvg. Sanity check.

La formula di aggregazione e' identica a OracleWeightedFedAvg:

    weight_k = alpha * (n_k / Sum n_j) + (1 - alpha) * urgency_k

con i pesi finali sottoposti a cap. La novita' rispetto a Oracle e' che
urgency_k puo' variare di round in round.

Time decay (Sezione 5 del report) implementato come moltiplicatore decrescente
applicato al boost di urgenza rispetto al peso uniforme. Default: 1.0 (inattivo).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
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


# ===========================================================================
# URGENCY PROVIDER: INTERFACCIA ASTRATTA
# ===========================================================================

class UrgencyProvider(ABC):
    """Sorgente del segnale di urgenza per client e per round.

    Tutte le sottoclassi devono implementare get_urgency, che restituisce
    un dict {client_name: score} con score in [0, +inf). Gli score NON
    devono essere normalizzati: la Strategy se ne occupa.
    """

    @abstractmethod
    def get_urgency(
        self,
        round_idx: int,
        client_names: list[str],
    ) -> dict[str, float]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome del provider, per logging e config snapshot."""
        ...


# ---------------------------------------------------------------------------
# PROVIDER 1: ORACLE (statico, derivato dalla ground truth esogena)
# ---------------------------------------------------------------------------

class OracleUrgencyProvider(UrgencyProvider):
    """Score di urgenza = quota della classe critica nel dataset del client.

    E' statico (non dipende dal round). Quando questo provider e' usato con
    NLPWeightedFedAvg, il risultato e' equivalente a OracleWeightedFedAvg.
    Serve come baseline per validare l'architettura.
    """

    def __init__(self, oracle_weights: dict[str, float]) -> None:
        if abs(sum(oracle_weights.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"oracle_weights non somma a 1: {sum(oracle_weights.values())}"
            )
        self.oracle_weights = oracle_weights

    def get_urgency(
        self,
        round_idx: int,
        client_names: list[str],
    ) -> dict[str, float]:
        return {c: self.oracle_weights[c] for c in client_names}

    @property
    def name(self) -> str:
        return "oracle"


# ---------------------------------------------------------------------------
# HELPER: caricamento pesi oracle dal manifest (stesso codice di
# oracle_weighted.py; lo duplichiamo per non creare dipendenze fra strategy)
# ---------------------------------------------------------------------------

def load_oracle_urgency(
    manifest_path: Path,
    critical_class: str,
    client_names: list[str],
) -> OracleUrgencyProvider:
    """Costruisce un OracleUrgencyProvider leggendo il global_manifest.json."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    crit_counts: dict[str, int] = {}
    for stats in manifest["per_country_stats"]:
        client = stats["client_name"]
        train = stats["train"]["boxes_per_class"].get(critical_class, 0)
        valid = stats["valid"]["boxes_per_class"].get(critical_class, 0)
        crit_counts[client] = train + valid

    missing = [c for c in client_names if c not in crit_counts]
    if missing:
        raise ValueError(
            f"Client non trovati nel manifest: {missing}. "
            f"Disponibili: {list(crit_counts.keys())}"
        )

    selected = {c: crit_counts[c] for c in client_names}
    total = sum(selected.values())
    if total == 0:
        raise ValueError(
            f"Nessuna bbox di classe critica '{critical_class}' nei client"
        )

    weights = {c: count / total for c, count in selected.items()}
    return OracleUrgencyProvider(weights)


# ===========================================================================
# STRATEGY
# ===========================================================================

class NLPWeightedFedAvg(FedAvg):
    """FedAvg con aggregazione pesata da un UrgencyProvider.

    Pattern di aggregazione:

        urgency_k = provider.get_urgency(round, clients)[k]
        urgency_normalized_k = urgency_k / Sum urgency_j
        fedavg_k = n_k / Sum n_j

        # time decay: il "boost" rispetto al peso uniforme decade nel tempo
        uniform = 1.0 / N
        decayed_urgency_k = uniform + decay(round) * (urgency_normalized_k - uniform)

        weight_k = alpha * fedavg_k + (1 - alpha) * decayed_urgency_k
        weight = apply_cap(weight)
    """

    def __init__(
        self,
        urgency_provider: UrgencyProvider,
        client_names: list[str],
        alpha: float = 0.5,
        cap: float = 0.5,
        time_decay: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            urgency_provider: oggetto che produce gli score di urgenza.
            client_names: ordine canonico dei client (idx 0..N-1).
            alpha: mix tra peso FedAvg classico e peso da urgenza. Default 0.5.
            cap: tetto massimo per il peso di un singolo client (in (0, 1]).
                 Default 0.5 (consigliato nel report).
            time_decay: fattore moltiplicativo per il boost di urgenza ad ogni
                 round. 1.0 = no decay, 0.9 = decade del 10% per round, ecc.
                 Default 1.0 (inattivo).
            kwargs: passati a FedAvg.__init__.
        """
        super().__init__(**kwargs)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha deve essere in [0, 1], dato: {alpha}")
        if not (0.0 < cap <= 1.0):
            raise ValueError(f"cap deve essere in (0, 1], dato: {cap}")
        if not (0.0 < time_decay <= 1.0):
            raise ValueError(
                f"time_decay deve essere in (0, 1], dato: {time_decay}"
            )

        self.provider = urgency_provider
        self.client_names = client_names
        self.alpha = alpha
        self.cap = cap
        self.time_decay = time_decay

        log(
            WARNING,
            "NLPWeightedFedAvg init: provider=%s, alpha=%.2f, cap=%.2f, time_decay=%.2f",
            urgency_provider.name, alpha, cap, time_decay,
        )

    # -- calcolo dei pesi di aggregazione --------------------------------

    def _compute_aggregation_weights(
        self,
        round_idx: int,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> tuple[list[float], list[str]]:
        """Calcola i pesi di aggregazione per il round corrente.

        Returns:
            Tupla (weights, client_names_in_order). Il secondo elemento serve
            per il logging: e' l'ordine dei client cosi' come compaiono in
            results, mappato sui nomi.
        """
        # Mappa cid -> client_name via partition_id passato in fit_metrics
        round_clients = [
            self.client_names[int(fit_res.metrics["partition_id"])]
            for _, fit_res in results
        ]

        # 1. Peso FedAvg classico (proporzionale a num_examples)
        n_k = np.array([fit_res.num_examples for _, fit_res in results],
                        dtype=float)
        n_total = n_k.sum()
        fedavg_w = (n_k / n_total) if n_total > 0 else np.ones(len(results)) / len(results)

        # 2. Score di urgenza dal provider, normalizzato sui soli client
        # partecipanti a questo round
        urgency_raw = self.provider.get_urgency(round_idx, round_clients)
        urgency_arr = np.array([urgency_raw[c] for c in round_clients], dtype=float)
        urgency_sum = urgency_arr.sum()
        if urgency_sum == 0:
            log(WARNING, "Provider %s ha tornato urgenza tutta zero, "
                "fallback su pesi uniformi", self.provider.name)
            urgency_norm = np.ones(len(results)) / len(results)
        else:
            urgency_norm = urgency_arr / urgency_sum

        # 3. Time decay sul "boost" rispetto al peso uniforme
        N = len(results)
        uniform = 1.0 / N
        decay_factor = self.time_decay ** (round_idx - 1)  # round 1 -> ^0 = 1.0
        decayed_urgency = uniform + decay_factor * (urgency_norm - uniform)

        # 4. Mix alpha
        weights = self.alpha * fedavg_w + (1 - self.alpha) * decayed_urgency

        # 5. Cap iterativo
        weights = self._apply_cap(weights)

        # 6. Sanity-rinormalizza per side-effects di floating point
        s = weights.sum()
        if abs(s - 1.0) > 1e-6:
            weights = weights / s

        return weights.tolist(), round_clients

    def _apply_cap(self, weights: np.ndarray) -> np.ndarray:
        """Cap iterativo: nessun peso oltre self.cap, eccesso ridistribuito."""
        weights = weights.copy()
        for _ in range(100):
            over = weights > self.cap
            if not over.any():
                return weights
            excess = (weights[over] - self.cap).sum()
            weights[over] = self.cap
            under = ~over
            if not under.any():
                break
            under_weights = weights[under]
            under_total = under_weights.sum()
            if under_total == 0:
                weights[under] = excess / under.sum()
            else:
                weights[under] += excess * (under_weights / under_total)
        return weights

    # -- override di aggregate_fit ---------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights, round_clients = self._compute_aggregation_weights(
            server_round, results
        )

        log(WARNING, "Round %d aggregation weights (provider=%s):",
            server_round, self.provider.name)
        for name, w in zip(round_clients, weights):
            log(WARNING, "  %s: %.4f", name, w)

        # Stesso trucco di OracleWeightedFedAvg: weights float -> num_examples int
        SCALE = 10_000
        weighted_results: list[tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), int(w * SCALE))
            for (_, fit_res), w in zip(results, weights)
        ]

        aggregated_ndarrays = aggregate(weighted_results)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics)
                            for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return aggregated_parameters, metrics_aggregated