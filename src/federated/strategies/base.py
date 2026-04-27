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
