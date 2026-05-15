"""
Smoke test centralizzato: YOLOv8n su RDD2022 Japan (Milestone 0).

Obiettivo: verificare che la pipeline (dataset YOLO + Ultralytics + GPU)
funzioni end-to-end. NON ci interessa la mAP finale, ci interessa che:
    1. il training parta senza errori,
    2. la loss scenda di epoch in epoch,
    3. la mAP@50 finale sia > 0 (anche bassa va bene).

Una volta validato lo smoke test, useremo questo script come riferimento
per costruire il client Flower nella Milestone 2.

Uso:
    python scripts/smoke_test_centralized.py
    python scripts/smoke_test_centralized.py --epochs 10 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "processed" / "japan_smoke" / "data.yaml"
RUNS_ROOT = PROJECT_ROOT / "experiments" / "smoke_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_YAML,
        help="Path al data.yaml di Ultralytics",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Modello base (yolov8n/s/m/l/x.pt). Default: nano",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device GPU ('0', '0,1', ...) o 'cpu'",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Numero di worker DataLoader. Su Windows tieni basso (2-4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        sys.exit(
            f"ERRORE: data.yaml non trovato in {args.data}.\n"
            f"Hai eseguito convert_voc_to_yolo_japan.py?"
        )

    # Import qui per evitare il caricamento di torch quando si chiede solo --help
    from ultralytics import YOLO

    print(f"Smoke test YOLOv8 - RDD2022 Japan")
    print(f"  data:    {args.data}")
    print(f"  model:   {args.model}")
    print(f"  epochs:  {args.epochs}")
    print(f"  batch:   {args.batch}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  device:  {args.device}")
    print(f"  seed:    {args.seed}")
    print(f"  output:  {RUNS_ROOT}")
    print("-" * 60)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = f"japan_yolov8n_{int(time.time())}"

    model = YOLO(args.model)

    # Ultralytics scarica yolov8n.pt automaticamente al primo uso (~6 MB).
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=False,
        # patience alta: vogliamo vedere tutte le epoch anche se non migliora
        patience=args.epochs + 1,
        # disabilitiamo plot pesanti per velocita'
        plots=True,
        verbose=True,
    )

    print("-" * 60)
    print("Training completato.")

    # Validazione finale esplicita su valid/ (Ultralytics la fa anche durante
    # il training, ma cosi' vediamo i numeri stampati a fine corsa)
    metrics = model.val(
        data=str(args.data),
        device=args.device,
        project=str(RUNS_ROOT),
        name=f"{run_name}_val",
        exist_ok=True,
        verbose=False,
    )

    print("\nMetriche finali su valid/:")
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    # Per-class: utilissimo per vedere se la classe critica (pothole, idx 3)
    # e' ancora il problema previsto
    print(f"\n  Per-class mAP@50:")
    class_names = ["D00", "D10", "D20", "D40"]
    for i, name in enumerate(class_names):
        marker = " <- CRITICA" if name == "D40" else ""
        try:
            ap50 = metrics.box.maps[i]
            print(f"    {name}: {ap50:.4f}{marker}")
        except (IndexError, AttributeError):
            print(f"    {name}: n/a{marker}")

    print(f"\nArtefatti salvati in: {RUNS_ROOT / run_name}")

    # Sanity check: smoke test "passato" se mAP@50 > 0
    if metrics.box.map50 > 0:
        print("\nSMOKE TEST PASSATO: il pipeline funziona end-to-end.")
        return 0
    else:
        print("\nSMOKE TEST FALLITO: mAP@50 = 0. Investigare.")
        return 1


if __name__ == "__main__":
    sys.exit(main())