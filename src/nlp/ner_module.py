"""
Stadio 1 della pipeline NLP: Named Entity Recognition (NER) geografico.

Obiettivo: da un testo di lamentela cittadina, estrarre le entità geografiche
(città, quartieri, strade) che permettono di localizzare il problema segnalato.

Tecnica: spaCy con modello inglese pre-addestrato en_core_web_lg, che espone
un NER già addestrato su categorie standard tra cui:
    - GPE (Geopolitical Entity): città, stati, nazioni (es. "Tokyo", "Japan")
    - LOC (Location): località non-politiche (es. "Shibuya crossing")
    - FAC (Facility): strade, edifici, aeroporti (es. "Main Street")

Motivazione: spaCy è la libreria standard NLP per Python; en_core_web_lg è
pre-addestrato su OntoNotes 5 con performance stato-dell'arte su NER inglese.
Non facciamo fine-tuning perché il modello pre-addestrato è sufficiente per
il nostro caso d'uso.

Uso:
    from src.nlp.ner_module import extract_geographic_entities
    entities = extract_geographic_entities("Pothole on Main Street in Tokyo")
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import spacy
from spacy.language import Language


# Categorie di entità considerate "geografiche" per il mapping (Stadio 2)
GEOGRAPHIC_LABELS = frozenset({"GPE", "LOC", "FAC"})

# Modello spaCy da usare. Installazione:
#     python3 -m spacy download en_core_web_lg
_SPACY_MODEL_NAME = "en_core_web_lg"


class GeographicEntity(TypedDict):
    """Struttura di un'entità geografica estratta dal testo."""
    text: str
    label: str
    start_char: int
    end_char: int


@lru_cache(maxsize=1)
def _load_spacy_model() -> Language:
    """Carica il modello spaCy una sola volta (cached)."""
    try:
        return spacy.load(_SPACY_MODEL_NAME)
    except OSError as e:
        raise RuntimeError(
            f"Modello spaCy '{_SPACY_MODEL_NAME}' non trovato. "
            f"Installa con: python3 -m spacy download {_SPACY_MODEL_NAME}"
        ) from e


def extract_geographic_entities(text: str) -> list[GeographicEntity]:
    """Estrae le entità geografiche da un testo di lamentela.

    Args:
        text: testo della lamentela in lingua inglese.

    Returns:
        Lista di entità geografiche trovate nell'ordine di apparizione.
        Vuota se nessuna entità rilevante è stata trovata.
    """
    if not text or not text.strip():
        return []

    nlp = _load_spacy_model()
    doc = nlp(text)

    entities: list[GeographicEntity] = []
    for ent in doc.ents:
        if ent.label_ in GEOGRAPHIC_LABELS:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return entities


def extract_primary_location(text: str) -> GeographicEntity | None:
    """Estrae la località principale del testo per il mapping FL.

    Strategia: priorità GPE (città) > LOC (località) > FAC (strade).
    Se ci sono più entità della stessa categoria, prende la prima trovata.

    Motivazione: nello Stadio 2 dobbiamo mappare una lamentela a UN client.
    Se il testo menziona sia "Main Street" (FAC) che "Tokyo" (GPE), è la
    città a determinare il client, non la strada.

    Args:
        text: testo della lamentela.

    Returns:
        L'entità geografica principale, o None se non trovata.
    """
    entities = extract_geographic_entities(text)
    if not entities:
        return None

    priority = {"GPE": 0, "LOC": 1, "FAC": 2}
    entities_sorted = sorted(
        entities,
        key=lambda e: (priority.get(e["label"], 99), e["start_char"]),
    )
    return entities_sorted[0]


# ============================================================================
# SMOKE TEST
# ============================================================================

def _run_smoke_test() -> None:
    """Esegue smoke test con: python3 -m src.nlp.ner_module"""
    test_cases = [
        ("There is a massive pothole on Main Street in Tokyo that damaged my car.", "Tokyo"),
        ("Terrible road conditions near Mumbai suburbs, especially Andheri.", "Mumbai"),
        ("Small crack on the highway near Prague, nothing urgent.", "Prague"),
        ("Reporting damage on the road.", None),
        ("The E6 highway in Norway is destroyed, huge potholes everywhere.", "Norway"),
    ]

    print("=" * 70)
    print("SMOKE TEST - ner_module.py")
    print("=" * 70)

    passed = 0
    for i, (text, expected) in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {text[:70]}...")
        entities = extract_geographic_entities(text)
        primary = extract_primary_location(text)

        print(f"  Entità totali: {len(entities)}")
        for ent in entities:
            print(f"    - {ent['text']!r} ({ent['label']})")
        print(f"  Primary location: {primary['text'] if primary else 'None'}")

        if expected is None:
            ok = primary is None
        else:
            ok = primary is not None and primary["text"] == expected

        if ok:
            print("  RESULT: PASS")
            passed += 1
        else:
            got = primary["text"] if primary else "None"
            print(f"  RESULT: FAIL (atteso: {expected!r}, ottenuto: {got!r})")

    print("\n" + "=" * 70)
    print(f"Risultato: {passed}/{len(test_cases)} test superati")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()