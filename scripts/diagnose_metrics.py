"""
Diagnostica delle metriche di Ultralytics: cosa contiene davvero metrics.box?

Prende un modello gia' addestrato (es. il best.pt della Milestone 0) e lancia
model.val() su un dataset, poi ispeziona TUTTI gli attributi del risultato
metrics.box per capire:
    - cosa sono r[i], p[i], maps[i], ap50, ap, ecc.
    - come sono indicizzate le classi
    - se le metriche per-class sono soglia-dipendenti

Output: tabella per-classe con tutte le metriche disponibili.
Da qui sapremo con certezza cosa estrarre come 'recall_critical'.

Uso:
    python scripts/diagnose_metrics.py
    python scripts/diagnose_metrics.py --weights path/to/best.pt --data path/to/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default: il modello che hai addestrato durante lo smoke test (M0).
# Adattalo se hai cancellato la run o se preferisci un best.pt piu' recente.
SMOKE_RUNS = PROJECT_ROOT / "experiments" / "smoke_tests"
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "processed" / "japan_smoke" / "data.yaml"
)


def find_latest_weights() -> Path | None:
    """Trova il piu' recente best.pt nelle run di smoke test."""
    if not SMOKE_RUNS.exists():
        return None
    candidates = list(SMOKE_RUNS.glob("*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path al .pt da valutare. Default: ultimo best.pt in smoke_tests/",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Path al data.yaml. Default: japan_smoke.",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=416)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    weights = args.weights or find_latest_weights()
    if weights is None or not weights.exists():
        sys.exit(
            "ERRORE: nessun checkpoint trovato.\n"
            f"  Provato: --weights {args.weights}\n"
            f"  Provato: ultimo best.pt in {SMOKE_RUNS}\n"
            "Passa --weights esplicitamente."
        )

    if not args.data.exists():
        sys.exit(f"ERRORE: data.yaml non trovato: {args.data}")

    from ultralytics import YOLO

    print(f"Diagnostica metriche Ultralytics")
    print(f"  weights: {weights}")
    print(f"  data:    {args.data}")
    print(f"  device:  {args.device}")
    print("-" * 70)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(args.data),
        device=args.device,
        imgsz=args.imgsz,
        verbose=False,
        plots=False,
    )

    print("\n=== ATTRIBUTI DI metrics.box ===")
    # Lista degli attributi non-privati di metrics.box
    box = metrics.box
    attrs = sorted([a for a in dir(box) if not a.startswith("_")])

    for attr in attrs:
        try:
            value = getattr(box, attr)
            # Skip metodi/funzioni
            if callable(value):
                continue
            # Rappresentazione compatta
            if hasattr(value, "shape"):
                print(f"  {attr:<20} shape={tuple(value.shape)}  "
                      f"values={value if value.size < 10 else '...'}")
            elif isinstance(value, (int, float)):
                print(f"  {attr:<20} = {value:.4f}")
            else:
                print(f"  {attr:<20} = {value}")
        except Exception as e:
            print(f"  {attr:<20} (errore: {e})")

    # Mapping noms classes
    print("\n=== MAPPING CLASSI ===")
    if hasattr(metrics, "names"):
        print(f"  metrics.names: {metrics.names}")
    if hasattr(model, "names"):
        print(f"  model.names:   {model.names}")

    # Tabella per-class delle metriche piu' importanti
    print("\n=== METRICHE PER-CLASS ===")
    class_names = ["D00", "D10", "D20", "D40"]
    print(f"{'class':<6} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 44)
    for i, name in enumerate(class_names):
        marker = " *" if name == "D40" else ""
        try:
            p = box.p[i] if hasattr(box, 'p') and len(box.p) > i else float('nan')
            r = box.r[i] if hasattr(box, 'r') and len(box.r) > i else float('nan')
            ap50 = (box.ap50[i] if hasattr(box, 'ap50') and len(box.ap50) > i
                    else float('nan'))
            ap = box.ap[i] if hasattr(box, 'ap') and len(box.ap) > i else float('nan')
            print(f"{name:<6} {p:>8.4f} {r:>8.4f} {ap50:>8.4f} {ap:>10.4f}{marker}")
        except Exception as e:
            print(f"{name:<6} errore: {e}")

    print("\nLegenda:")
    print("  P     = Precision per classe alla soglia di confidenza ottimale")
    print("  R     = Recall per classe alla soglia di confidenza ottimale")
    print("  mAP50 = Average Precision per classe a IoU=0.5")
    print("  mAP50-95 = AP medio per classe su IoU=0.5..0.95 (la 'mAP COCO')")
    print("\n* D40 = classe critica esogena (Componente A)")

    # Conferma di cosa stiamo aggregando in fl_run.py
    print("\n=== COSA AGGREGAVAMO IN fl_run.py ===")
    print(f"  Codice attuale: critical_recall = metrics.box.r[3]")
    print(f"  Valore qui:     {box.r[3] if len(box.r) > 3 else 'n/a'}")
    print("\n  Se questo numero oscilla nel tempo durante FL, il problema e' che\n"
          "  box.r e' Recall alla soglia di confidenza ottimale calcolata\n"
          "  da Ultralytics, che cambia di run in run. Per Recall stabile,\n"
          "  meglio usare AP@0.5 (box.ap50[3]).")

    return 0


if __name__ == "__main__":
    sys.exit(main())