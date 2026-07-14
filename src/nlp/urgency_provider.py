"""
Wrapper della pipeline NLP a 4 stadi per l'integrazione con il modulo FL.

Implementa l'interfaccia `UrgencyProvider` definita in
src.federated.strategies.nlp_weighted, in modo che il codice FL esistente
possa consumare l'output della pipeline NLP senza modifiche.

Design:
    All'inizializzazione, la classe processa il dataset di lamentele CSV
    attraverso tutti e 4 gli stadi:
        1. NER geografico  (extract_primary_location)
        2. Mapping location -> client FL  (resolve_client_from_text)
        3. Sentiment / severity  (compute_severity_score)
        4. Aggregazione per client  (aggregate_complaints_to_urgency)

    Il risultato è un vettore di urgenza pre-computato, che viene
    restituito a ogni chiamata di get_urgency (stateless rispetto al round).

    Il processing dello Stadio 3 è la parte più costosa (inferenza del
    Transformer). Lo eseguiamo una volta all'init, non ad ogni round.

Confronto con altri provider:
    - OracleUrgencyProvider: statico, letto dal manifest globale (ground truth
      sulla distribuzione della classe critica). Usato come upper bound.
    - NoisyOracleUrgencyProvider: aggiunge rumore controllato all'Oracle.
      Usato per la caratterizzazione di robustezza (RQ2).
    - NLPUrgencyProvider (questo): usa la pipeline NLP reale su dataset
      testuale sintetico. Dimostra l'integrazione end-to-end del modulo NLP
      con il modulo FL. Ha accuracy limitata dal fatto che il dataset di
      lamentele è piccolo e sintetico.

Uso:
    from src.nlp.urgency_provider import NLPUrgencyProvider

    provider = NLPUrgencyProvider(
        complaints_csv_path="data/nlp/synthetic_complaints.csv",
        all_clients=["japan", "india", "czech", "norway",
                     "united_states", "china_motorbike"],
    )

    urgency = provider.get_urgency(round_idx=0, client_names=[...])
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

# Importa l'ABC dalla stessa gerarchia usata dagli altri provider
from src.federated.strategies.nlp_weighted import UrgencyProvider

# Importa i moduli della pipeline NLP
from src.nlp.ner_module import extract_primary_location
from src.nlp.location_mapper import resolve_client_from_text
from src.nlp.sentiment_module import compute_severity_score
from src.nlp.aggregator import aggregate_complaints_to_urgency, Complaint


class NLPUrgencyProvider(UrgencyProvider):
    """UrgencyProvider basato sulla pipeline NLP reale.

    Processa un dataset CSV di lamentele attraverso i 4 stadi della pipeline
    (NER, mapping, sentiment, aggregazione) e restituisce il vettore di
    urgenza risultante ad ogni chiamata di get_urgency.

    Il processing avviene una sola volta all'inizializzazione (per efficienza),
    il che significa che l'urgenza è STATICA rispetto al round federato. Questo
    è coerente con il caso d'uso: le lamentele cittadine sono raccolte in
    background sul sistema, non ricalcolate ad ogni round di training.

    Per un deployment dinamico dove le lamentele arrivano in streaming, si
    potrebbe modificare la classe per accettare nuove lamentele fra round
    (self.add_complaint(text)) e ricalcolare l'urgenza incrementalmente.
    """

    def __init__(
        self,
        complaints_csv_path: str | Path,
        all_clients: list[str],
        verbose: bool = False,
    ) -> None:
        """
        Args:
            complaints_csv_path: path al CSV delle lamentele. Deve contenere
                almeno una colonna 'text' con il testo della lamentela in
                inglese.
            all_clients: lista dei nomi dei client FL. Il vettore di urgenza
                conterrà una entry per ognuno.
            verbose: se True, stampa progress e statistiche durante il
                processing.
        """
        self.complaints_csv_path = Path(complaints_csv_path)
        self.all_clients = list(all_clients)
        self.verbose = verbose

        if not self.complaints_csv_path.exists():
            raise FileNotFoundError(
                f"Dataset lamentele non trovato: {self.complaints_csv_path}"
            )

        # Esegui la pipeline sul dataset una volta sola
        self._processed_complaints: list[Complaint] = []
        self._urgency: dict[str, float] = {}
        self._process_dataset()

    @property
    def name(self) -> str:
        return "nlp_pipeline"

    def _process_dataset(self) -> None:
        """Esegue Stadio 1 + 2 + 3 + 4 sull'intero dataset."""
        if self.verbose:
            print(f"[{self.name}] Processing dataset da {self.complaints_csv_path}...")

        with self.complaints_csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                text = row.get("text", "").strip()
                if not text:
                    continue

                # Stadio 1: NER
                primary_entity = extract_primary_location(text)
                ner_entity_text = primary_entity["text"] if primary_entity else None

                # Stadio 2: Mapping location -> client (con fallback)
                client = resolve_client_from_text(text, ner_entity_text)

                # Se la lamentela non è mappabile a nessun client, la scartiamo
                # (non contribuisce all'urgenza di nessuno).
                if client is None:
                    if self.verbose:
                        print(f"  [{i}] UNMAPPED: {text[:60]}")
                    continue

                # Stadio 3: Sentiment / severity
                severity = compute_severity_score(text)

                self._processed_complaints.append({
                    "client": client,
                    "severity": severity,
                })

                if self.verbose and i % 10 == 0:
                    print(f"  processato {i}...")

        # Stadio 4: Aggregazione per client (produce dict normalizzato somma-a-1)
        self._urgency = aggregate_complaints_to_urgency(
            self._processed_complaints,
            self.all_clients,
        )

        if self.verbose:
            print(f"[{self.name}] Dataset processato:")
            print(f"  Lamentele totali processate: {len(self._processed_complaints)}")
            print(f"  Urgenza finale per client:")
            for c, u in sorted(self._urgency.items(), key=lambda x: -x[1]):
                n = sum(1 for x in self._processed_complaints if x["client"] == c)
                print(f"    {c:>18}: {u:.4f} ({n} lamentele)")

    def get_urgency(
        self,
        round_idx: int,
        client_names: list[str],
    ) -> dict[str, float]:
        """Restituisce il vettore di urgenza per i client specificati.

        Args:
            round_idx: numero del round federato corrente. Ignorato: la
                pipeline è statica rispetto al round (vedi docstring classe).
            client_names: sottoinsieme dei client per cui restituire l'urgenza.
                Utile se il round federato coinvolge solo un subset dei client.

        Returns:
            Dict {client_name: urgency_score}. Note: gli score NON sono
            normalizzati a somma 1 sul sotto-insieme richiesto (la Strategy
            se ne occupa, come da contratto UrgencyProvider).
        """
        # Restituisci l'urgenza pre-calcolata per i client richiesti.
        # NOTA: se un client_name richiesto NON è tra quelli configurati
        # all'init, ritorniamo 0 (comportamento difensivo, il codice della
        # Strategy non dovrebbe mai chiedere client sconosciuti).
        return {c: self._urgency.get(c, 0.0) for c in client_names}

    def get_complaint_stats(self) -> dict[str, dict[str, float]]:
        """Restituisce statistiche dettagliate per client (utile per debug/report).

        Returns:
            Dict con per ogni client: n_complaints, sum_severity, mean_severity,
            urgency finale (normalizzato).
        """
        from src.nlp.aggregator import summarize_by_client
        return summarize_by_client(self._processed_complaints, self.all_clients)


# ============================================================================
# SMOKE TEST
# ============================================================================

def _run_smoke_test() -> None:
    """Esegue smoke test con: python3 -m src.nlp.urgency_provider

    Attenzione: il test carica il modello Transformer di sentiment (~500MB
    la prima volta), quindi puo' richiedere qualche minuto al primo run.
    """
    from pathlib import Path

    csv_path = Path("data/nlp/synthetic_complaints_realistic.csv")
    if not csv_path.exists():
        print(f"ERRORE: file non trovato: {csv_path}")
        print("Esegui questo test dalla directory radice del progetto:")
        print("    python3 -m src.nlp.urgency_provider")
        return

    all_clients = [
        "japan", "india", "czech", "norway",
        "united_states", "china_motorbike",
    ]

    print("=" * 70)
    print("SMOKE TEST - NLPUrgencyProvider")
    print("=" * 70)

    provider = NLPUrgencyProvider(
        complaints_csv_path=csv_path,
        all_clients=all_clients,
        verbose=True,
    )

    # Test 1: get_urgency ritorna dict con le chiavi giuste
    print("\n[Test 1] get_urgency(0, all_clients):")
    urgency = provider.get_urgency(0, all_clients)
    assert set(urgency.keys()) == set(all_clients)
    for c, u in urgency.items():
        print(f"  {c:>18}: {u:.4f}")
    print(f"  Somma: {sum(urgency.values()):.6f} (atteso ~1.0)")

    # Test 2: get_urgency e' idempotente rispetto al round
    print("\n[Test 2] Verifica staticita' rispetto al round:")
    urgency_r0 = provider.get_urgency(0, all_clients)
    urgency_r5 = provider.get_urgency(5, all_clients)
    urgency_r99 = provider.get_urgency(99, all_clients)
    assert urgency_r0 == urgency_r5 == urgency_r99
    print(f"  round=0, 5, 99: risultato identico -> PASS")

    # Test 3: subset di client
    print("\n[Test 3] Subset di client:")
    subset = ["japan", "india"]
    urgency_sub = provider.get_urgency(0, subset)
    print(f"  richiesti: {subset}")
    print(f"  ottenuti:  {list(urgency_sub.keys())}")
    for c, u in urgency_sub.items():
        print(f"    {c}: {u:.4f}")
    assert set(urgency_sub.keys()) == set(subset)

    # Test 4: statistiche dettagliate
    print("\n[Test 4] Statistiche dettagliate per client:")
    stats = provider.get_complaint_stats()
    print(f"  {'client':>18} {'n':>4} {'sum':>7} {'mean':>7} {'urgency':>8}")
    for c in all_clients:
        s = stats[c]
        print(f"  {c:>18} {int(s['n_complaints']):>4} "
              f"{s['sum_severity']:>7.2f} {s['mean_severity']:>7.3f} "
              f"{s['urgency']:>8.4f}")

    print("\n" + "=" * 70)
    print(f"Provider name: {provider.name}")
    print("Tutti i test superati.")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()