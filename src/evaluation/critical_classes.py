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
