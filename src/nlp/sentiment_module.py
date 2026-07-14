"""
Stadio 3 della pipeline NLP: Sentiment/severity classification (v2).

Obiettivo: da un testo di lamentela cittadina, produrre uno score di urgenza
continuo in [0, 1] che quantifica quanto la lamentela è "grave".

Storia della progettazione (documentata per il report):
    v1 usava `nlptown/bert-multilingual-uncased-sentiment` (rating 1-5 stelle)
    con peso 1.0. Sul dataset sintetico ha ottenuto 63.3% di accuracy con un
    pattern di saturazione verso "high" su vocabolario tecnico e sotto-
    classificazione di lamentele formali gravi. Il modello, addestrato su
    recensioni di prodotti, non generalizza al dominio "public complaints".

    v2 (questa versione) adotta un'architettura ibrida ricalibrata:
    - Modello Transformer: `cardiffnlp/twitter-roberta-base-sentiment-latest`
      (addestrato su tweet, più vicino al tono complaint). Peso ridotto a 0.4.
    - Euristiche lessicali di dominio con peso aumentato: 0.6 complessivo,
      distribuito tra keyword boost, caps boost, exclamation boost e
      medium-priority indicator (nuovo).
    - Aggiunta gestione esplicita degli indicatori di priorità media
      (es. "medium priority", "medium severity", "within the month").

Modello scelto (v2):
    cardiffnlp/twitter-roberta-base-sentiment-latest
    - Output: negative / neutral / positive con confidence
    - Addestrato su tweet (~124M), dominio "opinioni brevi"
    - Dimensione ~500 MB, gira comodamente su CPU
    Riferimento: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

Uso:
    from src.nlp.sentiment_module import compute_severity_score
    severity = compute_severity_score("HUGE pothole! Someone will die!")
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TypedDict

from transformers import pipeline


# ============================================================================
# CONFIGURAZIONE MODELLO
# ============================================================================

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Il modello Twitter-RoBERTa ritorna label "negative", "neutral", "positive"
# insieme a un score di confidence. Mappiamo (label, score) a un valore in
# [0, 1] rappresentante la severity implicita del testo.
# Logica: negative -> alta severity; positive -> bassa severity; neutral -> 0.5.
def _map_sentiment_to_severity(label: str, score: float) -> float:
    label = label.lower().strip()
    if label == "negative":
        # negative con alta confidence -> alta severity
        return 0.5 + 0.5 * score
    elif label == "positive":
        # positive con alta confidence -> bassa severity
        return 0.5 - 0.5 * score
    else:
        # neutral -> centro
        return 0.5


# ============================================================================
# EURISTICHE LESSICALI (specifiche del dominio "road complaints")
# ============================================================================

# Parole/frasi chiave che indicano ALTA urgenza
_HIGH_URGENCY_KEYWORDS = frozenset({
    "urgent", "urgently", "immediate", "immediately",
    "dangerous", "danger", "hazard", "hazardous",
    "critical", "emergency", "asap",
    "die", "death", "fatal", "fatality", "killed",
    "life-threatening", "life threatening",
    "accident", "accidents", "crashed", "crashing", "crash",
    "injured", "injury", "injuries", "hurt",
    "massive", "enormous", "huge", "deep",
    "destroyed", "destruction", "disaster", "catastrophe",
    "must act", "must be fixed", "fix now", "act now",
    "someone will",
    "furious",  # aggiunto da analisi dataset
})

# Parole/frasi chiave che indicano BASSA urgenza
_LOW_URGENCY_KEYWORDS = frozenset({
    "minor", "small", "slight", "little",
    "not urgent", "not critical", "nothing urgent",
    "not major", "not critical yet", "monitor",
    "log for", "log it", "future reference",
    "low priority", "low severity",
    "not much", "nothing critical",
    "note it",
    "for reference",
})

# NUOVO in v2: indicatori espliciti di priorità MEDIA. Il modello v1 non
# aveva un modo per catturare questo segnale, che è invece esplicito nel
# nostro dataset.
_MEDIUM_URGENCY_KEYWORDS = frozenset({
    "medium priority", "medium severity", "medium urgency",
    "within the month", "in the next few weeks", "next two weeks",
    "coming weeks", "next week",
    "should be repaired", "should be filled", "would benefit",
    "plan maintenance", "monitor closely",
})


class SeverityBreakdown(TypedDict):
    """Componenti dello score di severità (utile per debug/analisi)."""
    severity: float
    transformer_severity: float
    transformer_weight: float
    keyword_high: float
    keyword_low: float
    keyword_medium_pull: float
    caps_boost: float
    exclamation_boost: float


# ============================================================================
# PESI (v2: Transformer 0.4, euristiche 0.6 complessivo)
# ============================================================================

_TRANSFORMER_WEIGHT = 0.4

# Boost massimi delle euristiche (somma teorica <= 0.6)
_KW_HIGH_MAX = 0.30
_KW_LOW_MAX = -0.40
_KW_MEDIUM_PULL_MAX = 0.15   # attira verso il centro (0.5)
_CAPS_MAX = 0.15
_EXCL_MAX = 0.10


# ============================================================================
# CARICAMENTO MODELLO
# ============================================================================

@lru_cache(maxsize=1)
def _load_sentiment_pipeline():
    """Carica la pipeline di sentiment una sola volta (cached)."""
    return pipeline(
        task="sentiment-analysis",
        model=_MODEL_NAME,
        tokenizer=_MODEL_NAME,
        device=-1,
        truncation=True,
        max_length=512,
    )


# ============================================================================
# COMPONENTI DELLO SCORE
# ============================================================================

def _transformer_severity(text: str) -> float:
    """Componente Transformer: severity implicita dal sentiment."""
    pipe = _load_sentiment_pipeline()
    result = pipe(text)[0]
    return _map_sentiment_to_severity(result["label"], float(result["score"]))


def _count_hits(text_lower: str, keywords: frozenset) -> int:
    """Conta le occorrenze di keyword nel testo (saturato a 3)."""
    return min(sum(1 for kw in keywords if kw in text_lower), 3)


def _keyword_boosts(text: str) -> tuple[float, float, float]:
    """Calcola i tre boost lessicali (high, low, medium-pull).

    Returns:
        (kw_high_boost, kw_low_boost, kw_medium_pull)
    """
    text_lower = text.lower()
    high_hits = _count_hits(text_lower, _HIGH_URGENCY_KEYWORDS)
    low_hits = _count_hits(text_lower, _LOW_URGENCY_KEYWORDS)
    medium_hits = _count_hits(text_lower, _MEDIUM_URGENCY_KEYWORDS)

    kw_high = _KW_HIGH_MAX * (high_hits / 3)
    kw_low = _KW_LOW_MAX * (low_hits / 3)   # negativo
    kw_medium_pull = _KW_MEDIUM_PULL_MAX * (medium_hits / 3)   # per pull-to-center

    return kw_high, kw_low, kw_medium_pull


def _caps_boost(text: str) -> float:
    words = re.findall(r"\b[A-Z]{3,}\b", text)
    return min(_CAPS_MAX * (len(words) / 3), _CAPS_MAX)


def _exclamation_boost(text: str) -> float:
    n_excl = text.count("!")
    if n_excl == 0:
        return 0.0
    return min(_EXCL_MAX * (n_excl / 3), _EXCL_MAX)


# ============================================================================
# API PRIMARIA
# ============================================================================

def compute_severity_score(text: str) -> float:
    """Calcola lo score di urgenza di una lamentela.

    Formula finale (v2):
        base = 0.4 * transformer_severity
        boost_urgency = kw_high + kw_low (in [-0.3, +0.3])
        boost_caps = caps_boost (in [0, 0.15])
        boost_excl = exclamation_boost (in [0, 0.10])

        # Medium pull: attira lo score verso 0.5 se presente
        medium_pull_amount = kw_medium_pull (in [0, 0.15])
        pull_direction = (0.5 - current_estimate)

        final = base + boost_urgency + boost_caps + boost_excl
              + medium_pull_amount * pull_direction

        # Aggiunta della componente uniforme che porta il transformer
        # da peso 0.4 al centro senza il boost:
        final += (1 - 0.4) * 0.5    (contribution neutra base)

    Il risultato è clippato in [0, 1].

    Args:
        text: testo della lamentela.

    Returns:
        Score di urgenza in [0, 1].
    """
    return compute_severity_breakdown(text)["severity"]


def compute_severity_breakdown(text: str) -> SeverityBreakdown:
    """Come compute_severity_score, ma ritorna anche i componenti."""
    if not text or not text.strip():
        return {
            "severity": 0.0,
            "transformer_severity": 0.0,
            "transformer_weight": _TRANSFORMER_WEIGHT,
            "keyword_high": 0.0,
            "keyword_low": 0.0,
            "keyword_medium_pull": 0.0,
            "caps_boost": 0.0,
            "exclamation_boost": 0.0,
        }

    t_sev = _transformer_severity(text)
    kw_high, kw_low, kw_medium_pull = _keyword_boosts(text)
    c_boost = _caps_boost(text)
    e_boost = _exclamation_boost(text)

    # Formula:
    # base contribution del Transformer con peso 0.4:
    transformer_contrib = _TRANSFORMER_WEIGHT * t_sev

    # componente neutra per la parte "non coperta" dal Transformer (peso 0.6):
    # centrata su 0.5 per non biasare
    neutral_contrib = (1 - _TRANSFORMER_WEIGHT) * 0.5

    # Boost:
    boosts = kw_high + kw_low + c_boost + e_boost

    # Preliminare (senza medium pull):
    preliminary = transformer_contrib + neutral_contrib + boosts

    # Medium pull: attira verso il centro (0.5) proporzionalmente a kw_medium_pull
    # Se preliminary è lontano da 0.5, il pull ha effetto maggiore.
    pull_direction = 0.5 - preliminary
    medium_pull_effect = kw_medium_pull * pull_direction
    final = preliminary + medium_pull_effect

    final = max(0.0, min(1.0, final))

    return {
        "severity": final,
        "transformer_severity": t_sev,
        "transformer_weight": _TRANSFORMER_WEIGHT,
        "keyword_high": kw_high,
        "keyword_low": kw_low,
        "keyword_medium_pull": kw_medium_pull,
        "caps_boost": c_boost,
        "exclamation_boost": e_boost,
    }


def severity_to_category(severity: float) -> str:
    """Converte score continuo a categoria discreta (low/medium/high)."""
    if severity < 0.4:
        return "low"
    elif severity < 0.7:
        return "medium"
    else:
        return "high"


# ============================================================================
# SMOKE TEST
# ============================================================================

def _run_smoke_test() -> None:
    """Esegue smoke test con: python3 -m src.nlp.sentiment_module"""
    test_cases = [
        ("MASSIVE pothole! Someone will die if this isn't fixed! URGENT!!", "high"),
        ("Life-threatening road damage. Multiple accidents this week. Immediate action needed!", "high"),
        ("Small surface wear near the station. Log for future reference.", "low"),
        ("Minor crack, not urgent, monitor periodically.", "low"),
        ("Reporting alligator cracking on the highway. Medium priority repair.", "medium"),
        ("Growing pothole should be repaired within the month.", "medium"),
    ]

    print("=" * 70)
    print("SMOKE TEST - sentiment_module.py (v2)")
    print(f"Modello: {_MODEL_NAME}")
    print("=" * 70)

    passed = 0
    for i, (text, expected_cat) in enumerate(test_cases, 1):
        breakdown = compute_severity_breakdown(text)
        got_cat = severity_to_category(breakdown["severity"])
        ok = got_cat == expected_cat

        print(f"\n[Test {i}] Expected: {expected_cat}, Got: {got_cat} - {'PASS' if ok else 'FAIL'}")
        print(f"  Text: {text[:75]}")
        print(f"  Transformer sev: {breakdown['transformer_severity']:.2f} "
              f"(weight {breakdown['transformer_weight']})")
        print(f"  Keyword high: +{breakdown['keyword_high']:.2f}")
        print(f"  Keyword low:  {breakdown['keyword_low']:+.2f}")
        print(f"  Keyword medium pull: {breakdown['keyword_medium_pull']:.2f}")
        print(f"  Caps boost: +{breakdown['caps_boost']:.2f}")
        print(f"  Exclamation boost: +{breakdown['exclamation_boost']:.2f}")
        print(f"  FINAL: {breakdown['severity']:.3f} -> {got_cat}")

        if ok:
            passed += 1

    print("\n" + "=" * 70)
    print(f"Risultato: {passed}/{len(test_cases)} test superati")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()