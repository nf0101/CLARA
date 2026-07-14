"""
Stadio 2 della pipeline NLP: mapping location -> client FL.

Obiettivo: data un'entità geografica estratta dallo Stadio 1 (es. "Tokyo",
"Shibuya", "E6 highway"), determinare a quale client del sistema Federated
Learning va assegnata la lamentela corrispondente.

Nel presente lavoro i client FL corrispondono a paesi (partizionamento
country-wise del dataset RDD2022). Il mapping è quindi a granularità paese:
città o strade note vengono mappate al paese di appartenenza.

Il modulo espone anche una funzione di FALLBACK pattern-based che ricerca
città note direttamente nel testo grezzo della lamentela, utile quando il
NER neurale non estrae alcuna entità (es. per città meno frequenti nel
corpus di training del modello).

Estensione a deployment reale (intra-citta'):
    In un deployment reale su singola città, i client corrisponderebbero
    a veicoli o zone di pattugliamento. Il mapping location -> client
    userebbe un servizio di geocoding (es. OpenStreetMap Nominatim) e
    poligoni delle zone. L'architettura della pipeline resta invariata:
    cambia solo l'implementazione di map_location_to_client(); gli altri
    stadi sono agnostici rispetto alla scala geografica.

Uso:
    from src.nlp.location_mapper import (
        map_location_to_client,
        map_location_to_client_fuzzy,
        fallback_search_client_in_text,
    )
    client = map_location_to_client("Tokyo")
    client = map_location_to_client_fuzzy("Shibuya crossing")
    client = fallback_search_client_in_text("Ostrava has huge potholes")
"""

from __future__ import annotations


# ============================================================================
# TABELLA DI MAPPING
# ============================================================================
#
# Le chiavi sono in lowercase per il matching case-insensitive.
# Le città sono state selezionate coerentemente con il dataset RDD2022
# (che ha immagini raccolte da queste regioni geografiche) e con il dataset
# sintetico di lamentele in data/nlp/synthetic_complaints.csv.

_LOCATION_TO_CLIENT: dict[str, str] = {
    # --- Japan ---
    "tokyo": "japan",
    "osaka": "japan",
    "kyoto": "japan",
    "yokohama": "japan",
    "nagoya": "japan",
    "sapporo": "japan",
    "fukuoka": "japan",
    "sendai": "japan",
    "shibuya": "japan",
    "ginza": "japan",
    "japan": "japan",

    # --- India ---
    "mumbai": "india",
    "delhi": "india",
    "bangalore": "india",
    "chennai": "india",
    "kolkata": "india",
    "pune": "india",
    "hyderabad": "india",
    "ahmedabad": "india",
    "jaipur": "india",
    "lucknow": "india",
    "india": "india",

    # --- Czech Republic ---
    "prague": "czech",
    "brno": "czech",
    "ostrava": "czech",
    "plzen": "czech",
    "cesky krumlov": "czech",
    "olomouc": "czech",
    "karlovy vary": "czech",
    "liberec": "czech",
    "hradec kralove": "czech",
    "czech republic": "czech",
    "czechia": "czech",

    # --- Norway ---
    "oslo": "norway",
    "bergen": "norway",
    "trondheim": "norway",
    "stavanger": "norway",
    "kristiansand": "norway",
    "tromso": "norway",
    "alesund": "norway",
    "drammen": "norway",
    "fredrikstad": "norway",
    "sandnes": "norway",
    "norway": "norway",

    # --- United States ---
    "new york": "united_states",
    "new york city": "united_states",
    "los angeles": "united_states",
    "chicago": "united_states",
    "houston": "united_states",
    "san francisco": "united_states",
    "phoenix": "united_states",
    "boston": "united_states",
    "seattle": "united_states",
    "miami": "united_states",
    "denver": "united_states",
    "united states": "united_states",
    "usa": "united_states",
    "u.s.": "united_states",

    # --- China (motorbike subset) ---
    "beijing": "china_motorbike",
    "shanghai": "china_motorbike",
    "guangzhou": "china_motorbike",
    "shenzhen": "china_motorbike",
    "chengdu": "china_motorbike",
    "hangzhou": "china_motorbike",
    "xi'an": "china_motorbike",
    "wuhan": "china_motorbike",
    "nanjing": "china_motorbike",
    "chongqing": "china_motorbike",
    "china": "china_motorbike",
}


# Lista canonica dei nomi dei client FL, per validazione consistenza
KNOWN_CLIENTS: frozenset[str] = frozenset({
    "japan", "india", "czech", "norway", "united_states", "china_motorbike",
})

# Sanity check statico: tutti i mapping puntano a client validi
assert set(_LOCATION_TO_CLIENT.values()) <= KNOWN_CLIENTS, (
    "location_mapper contiene mapping a client sconosciuti"
)


# ============================================================================
# API PRIMARIA (funzioni per input già "puliti" da spaCy)
# ============================================================================

def map_location_to_client(location: str) -> str | None:
    """Mappa un nome di località al client FL corrispondente.

    Il matching è case-insensitive e ignora spazi extra. Fa un lookup
    esatto sulla tabella. Se non trova match, ritorna None.

    Args:
        location: nome della località (es. "Tokyo", "Shibuya crossing").

    Returns:
        Nome del client FL (es. "japan"), o None se non riconosciuto.
    """
    if not location:
        return None
    key = location.strip().lower()
    return _LOCATION_TO_CLIENT.get(key)


def map_location_to_client_fuzzy(location: str) -> str | None:
    """Variante con fuzzy matching: cerca sottostringhe.

    Utile per location come "Shibuya crossing" o "New York City area" dove
    la parte informativa (Shibuya, New York) è dentro una stringa più lunga.

    Args:
        location: nome della località, eventualmente con parole extra.

    Returns:
        Nome del client FL, o None se non trova match.
    """
    if not location:
        return None

    key = location.strip().lower()

    # 1. Match esatto (via)
    if key in _LOCATION_TO_CLIENT:
        return _LOCATION_TO_CLIENT[key]

    # 2. Match parziale: verifica se una entità nota è sottostringa di key
    for known_loc, client in _LOCATION_TO_CLIENT.items():
        if known_loc in key or key in known_loc:
            return client

    return None


# ============================================================================
# FALLBACK: pattern-match sul testo grezzo
# ============================================================================

def fallback_search_client_in_text(text: str) -> str | None:
    """Fallback: cerca città note direttamente nel testo della lamentela.

    Usato quando lo Stadio 1 (NER neurale) non estrae alcuna entità
    geografica, o quando l'entità estratta non è mappabile. È una strategia
    di robustezza ampiamente adottata nei sistemi di produzione NLP per
    gestire nomi propri fuori distribuzione (out-of-vocabulary) rispetto al
    corpus di training del modello NER.

    Strategia: ricerca sottostringa case-insensitive delle chiavi della
    tabella `_LOCATION_TO_CLIENT` nel testo. Se trova più match, ritorna
    quello più lungo (assunzione: la corrispondenza più specifica è la più
    informativa; es. "new york city" prima di "new york").

    Nota: non gestisce ambiguità semantiche (es. "Portland" esiste in USA e
    Australia). Nel presente dataset non abbiamo casi ambigui perché il
    dominio è ristretto ai 6 paesi del challenge CRDDC.

    Args:
        text: testo grezzo della lamentela.

    Returns:
        Nome del client FL, o None se nessuna città nota è stata trovata.

    Example:
        >>> fallback_search_client_in_text("Ostrava industrial zone has huge potholes")
        'czech'
        >>> fallback_search_client_in_text("Drammen main street shows some wear")
        'norway'
        >>> fallback_search_client_in_text("No city mentioned here")
        None
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # Colleziona tutti i match, ordinati per lunghezza decrescente della chiave
    # (le corrispondenze più lunghe/specifiche hanno priorità).
    matches: list[tuple[int, str]] = []   # (lunghezza_chiave, client)
    for known_loc, client in _LOCATION_TO_CLIENT.items():
        if known_loc in text_lower:
            matches.append((len(known_loc), client))

    if not matches:
        return None

    # Ritorna il client corrispondente al match più lungo
    matches.sort(key=lambda m: m[0], reverse=True)
    return matches[0][1]


# ============================================================================
# API COMBINATA (funzione di alto livello per uso end-to-end)
# ============================================================================

def resolve_client_from_text(text: str, ner_entity: str | None) -> str | None:
    """Risolve il client FL a partire dal testo, con strategia ibrida.

    Combina:
    - map_location_to_client_fuzzy() sull'entità NER (se presente): tenta
      il mapping semantico via NER neurale.
    - fallback_search_client_in_text() sul testo grezzo (se il primo fallisce):
      cerca città note direttamente nel testo come pattern-match.

    Questa strategia ibrida è una best practice consolidata nei sistemi di
    produzione NLP: il modello neurale ha alta precisione ma bassa copertura
    sui nomi propri fuori distribuzione; il pattern-match ha alta copertura
    ma nessuna comprensione semantica. La combinazione dei due massimizza
    la robustezza.

    Args:
        text: testo grezzo della lamentela.
        ner_entity: entità geografica estratta dallo Stadio 1 (può essere None
                    se il NER non ha trovato nulla).

    Returns:
        Nome del client FL, o None se nessuna strategia ha successo.
    """
    # Strategia 1: mapping dell'entità NER
    if ner_entity:
        client = map_location_to_client_fuzzy(ner_entity)
        if client is not None:
            return client

    # Strategia 2: fallback pattern-match sul testo grezzo
    return fallback_search_client_in_text(text)


# ============================================================================
# SMOKE TEST
# ============================================================================

def _run_smoke_test() -> None:
    """Esegue smoke test con: python3 -m src.nlp.location_mapper"""
    exact_cases = [
        ("Tokyo", "japan"),
        ("tokyo", "japan"),
        ("  Tokyo  ", "japan"),
        ("Shibuya", "japan"),
        ("Mumbai", "india"),
        ("New York", "united_states"),
        ("Prague", "czech"),
        ("Norway", "norway"),
        ("Beijing", "china_motorbike"),
        ("Unknown Place", None),
        ("", None),
    ]

    fuzzy_cases = [
        ("Shibuya crossing", "japan"),
        ("New York City suburbs", "united_states"),
        ("Mumbai downtown area", "india"),
        ("Downtown Prague old town", "czech"),
        ("Somewhere unknown", None),
    ]

    fallback_cases = [
        # Casi presi dai 5 errori originali della fase end-to-end
        ("Ostrava industrial zone road is heavily damaged with deep potholes.", "czech"),
        ("Historic city center in Cesky Krumlov has damaged cobblestones.", "czech"),
        ("Drammen main street shows some surface wear.", "norway"),
        ("Fredrikstad bridge access road has developed dangerous potholes.", "norway"),
        ("Sandnes industrial area roads have minor cracks.", "norway"),
        # Contro-esempio: testo senza località note
        ("The road is damaged and needs urgent attention.", None),
    ]

    print("=" * 70)
    print("SMOKE TEST - location_mapper.py")
    print("=" * 70)

    print("\n[1/3] Exact match:")
    passed = 0
    for loc, expected in exact_cases:
        got = map_location_to_client(loc)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {loc!r} -> {got!r} (atteso: {expected!r})")
        if ok:
            passed += 1
    print(f"  Risultato: {passed}/{len(exact_cases)}")

    print("\n[2/3] Fuzzy match:")
    passed_fuzzy = 0
    for loc, expected in fuzzy_cases:
        got = map_location_to_client_fuzzy(loc)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {loc!r} -> {got!r} (atteso: {expected!r})")
        if ok:
            passed_fuzzy += 1
    print(f"  Risultato: {passed_fuzzy}/{len(fuzzy_cases)}")

    print("\n[3/3] Fallback pattern-match (testo grezzo):")
    passed_fb = 0
    for text, expected in fallback_cases:
        got = fallback_search_client_in_text(text)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {text[:60]!r}... -> {got!r} (atteso: {expected!r})")
        if ok:
            passed_fb += 1
    print(f"  Risultato: {passed_fb}/{len(fallback_cases)}")

    print("\n" + "=" * 70)
    tot = passed + passed_fuzzy + passed_fb
    tot_all = len(exact_cases) + len(fuzzy_cases) + len(fallback_cases)
    print(f"Totale: {tot}/{tot_all}")
    print(f"Locations in tabella: {len(_LOCATION_TO_CLIENT)}")
    print(f"Client FL: {sorted(KNOWN_CLIENTS)}")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()