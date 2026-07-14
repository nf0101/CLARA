"""
Stadio 4 della pipeline NLP: aggregazione per client.

Obiettivo: dato un insieme di lamentele con severity e client assegnato,
produrre un vettore di urgenze per client `{client_name: urgency}` normalizzato
a somma 1, in un formato compatibile con l'interfaccia `UrgencyProvider` del
modulo FL.

Formula di aggregazione:
    raw_urgency(client) = sum(severity_i per ogni lamentela i assegnata al client)

    Questa formula scala naturalmente con il "volume × intensità": un client
    con molte lamentele high avrà raw_urgency alto, e un client con nessuna
    lamentela avrà raw_urgency zero. Dopo normalizzazione a somma 1, il
    risultato è direttamente compatibile con OracleUrgencyProvider.

Gestione dei client senza lamentele:
    Se un client non ha nessuna lamentela nel batch analizzato, riceve
    un'urgenza minima positiva `MIN_URGENCY` per evitare che venga totalmente
    ignorato dall'aggregazione FL (il pattern è comune nei sistemi reali:
    "assenza di segnalazioni != assenza di problemi").

Uso:
    from src.nlp.aggregator import aggregate_complaints_to_urgency

    complaints = [
        {"client": "japan", "severity": 0.9},
        {"client": "japan", "severity": 0.3},
        {"client": "india", "severity": 0.8},
        # ...
    ]
    urgency = aggregate_complaints_to_urgency(
        complaints,
        all_clients=["japan", "india", "czech", "norway", "united_states", "china_motorbike"],
    )
    # -> {"japan": 0.22, "india": 0.18, "czech": 0.02, ...}   # somma a 1
"""

from __future__ import annotations

from typing import TypedDict


# Urgenza minima per client senza lamentele (evita divisione per zero e
# esclusione totale dall'aggregazione FL).
MIN_URGENCY = 0.01


class Complaint(TypedDict):
    """Struttura di una lamentela già processata dagli Stadi 1-3."""
    client: str        # nome del client FL (dall'output dello Stadio 2)
    severity: float    # score di urgenza in [0, 1] (dall'output dello Stadio 3)


def aggregate_complaints_to_urgency(
    complaints: list[Complaint],
    all_clients: list[str],
) -> dict[str, float]:
    """Aggrega lamentele in un vettore di urgenza per client normalizzato.

    Args:
        complaints: lista di lamentele, ognuna con 'client' e 'severity'.
                    Lamentele con client=None o non in all_clients vengono
                    ignorate (non contribuiscono a nessuna urgenza).
        all_clients: lista completa dei client FL. Il risultato conterrà
                     una chiave per ognuno di questi client.

    Returns:
        Dict `{client_name: urgency}` con:
        - Una chiave per ogni client in all_clients.
        - Valori in [MIN_URGENCY, 1].
        - Somma dei valori = 1.0 (a meno di errori di floating point).

    Example:
        >>> complaints = [
        ...     {"client": "japan", "severity": 0.9},
        ...     {"client": "japan", "severity": 0.7},
        ...     {"client": "india", "severity": 0.5},
        ... ]
        >>> urgency = aggregate_complaints_to_urgency(
        ...     complaints,
        ...     all_clients=["japan", "india", "czech"]
        ... )
        >>> # japan_raw = 1.6, india_raw = 0.5, czech_raw = 0.01
        >>> # normalizzato: japan=0.76, india=0.24, czech=0.005 (circa)
    """
    if not all_clients:
        raise ValueError("all_clients non puo' essere vuoto")

    # 1. Somma severity per client
    raw_urgency: dict[str, float] = {c: 0.0 for c in all_clients}
    for complaint in complaints:
        client = complaint.get("client")
        severity = complaint.get("severity", 0.0)
        if client in raw_urgency:
            raw_urgency[client] += severity

    # 2. Applica minimo per evitare zero
    for c in all_clients:
        if raw_urgency[c] == 0.0:
            raw_urgency[c] = MIN_URGENCY

    # 3. Normalizza a somma 1
    total = sum(raw_urgency.values())
    if total <= 0:
        # Fallback difensivo: uniforme
        return {c: 1.0 / len(all_clients) for c in all_clients}

    return {c: raw_urgency[c] / total for c in all_clients}


def summarize_by_client(
    complaints: list[Complaint],
    all_clients: list[str],
) -> dict[str, dict[str, float]]:
    """Ritorna statistiche dettagliate per client (utile per debug/report).

    Args:
        complaints: lista di lamentele.
        all_clients: lista completa dei client FL.

    Returns:
        Dict `{client_name: {n_complaints, sum_severity, mean_severity, urgency}}`.
    """
    stats: dict[str, dict[str, float]] = {}
    for c in all_clients:
        client_complaints = [x for x in complaints if x.get("client") == c]
        n = len(client_complaints)
        sum_sev = sum(x.get("severity", 0.0) for x in client_complaints)
        mean_sev = (sum_sev / n) if n > 0 else 0.0
        stats[c] = {
            "n_complaints": float(n),
            "sum_severity": sum_sev,
            "mean_severity": mean_sev,
        }

    urgency = aggregate_complaints_to_urgency(complaints, all_clients)
    for c in all_clients:
        stats[c]["urgency"] = urgency[c]

    return stats


# ============================================================================
# SMOKE TEST
# ============================================================================

def _run_smoke_test() -> None:
    """Esegue smoke test con: python3 -m src.nlp.aggregator"""
    all_clients = ["japan", "india", "czech", "norway", "united_states", "china_motorbike"]

    # Test 1: distribuzione uniforme se ogni client ha 1 lamentela severity 0.5
    complaints_uniform = [
        {"client": c, "severity": 0.5} for c in all_clients
    ]
    urgency = aggregate_complaints_to_urgency(complaints_uniform, all_clients)
    print("[Test 1] Uniforme (ogni client 1 lamentela severity 0.5):")
    for c, u in urgency.items():
        print(f"  {c:>18}: {u:.4f}")
    print(f"  Somma: {sum(urgency.values()):.6f}")
    print(f"  Atteso: ~1/6 = {1/6:.4f} per ognuno")

    # Test 2: japan e india dominano
    complaints_biased = [
        {"client": "japan", "severity": 1.0},
        {"client": "japan", "severity": 0.9},
        {"client": "japan", "severity": 0.8},
        {"client": "india", "severity": 0.9},
        {"client": "india", "severity": 0.7},
        {"client": "czech", "severity": 0.3},
    ]
    urgency = aggregate_complaints_to_urgency(complaints_biased, all_clients)
    print("\n[Test 2] Bias verso japan/india:")
    for c, u in urgency.items():
        print(f"  {c:>18}: {u:.4f}")
    print(f"  Somma: {sum(urgency.values()):.6f}")
    print(f"  Atteso: japan > india > czech > norway/us/china (=MIN)")

    # Test 3: nessuna lamentela (edge case)
    urgency_empty = aggregate_complaints_to_urgency([], all_clients)
    print("\n[Test 3] Nessuna lamentela:")
    for c, u in urgency_empty.items():
        print(f"  {c:>18}: {u:.4f}")
    print(f"  Somma: {sum(urgency_empty.values()):.6f}")
    print(f"  Atteso: uniforme ~{1/6:.4f} per ognuno")

    # Test 4: verifica summarize_by_client
    print("\n[Test 4] summarize_by_client sul bias di Test 2:")
    stats = summarize_by_client(complaints_biased, all_clients)
    print(f"  {'client':>18} {'n':>4} {'sum':>7} {'mean':>7} {'urgency':>8}")
    for c, s in stats.items():
        print(f"  {c:>18} {int(s['n_complaints']):>4} "
              f"{s['sum_severity']:>7.2f} {s['mean_severity']:>7.3f} {s['urgency']:>8.4f}")

    # Test 5: sanity check somma-a-1
    assert abs(sum(urgency.values()) - 1.0) < 1e-6, "urgency non somma a 1!"
    assert abs(sum(urgency_empty.values()) - 1.0) < 1e-6, "urgency_empty non somma a 1!"
    print("\n[Test 5] Sanity check somma-a-1: PASS")


if __name__ == "__main__":
    _run_smoke_test()