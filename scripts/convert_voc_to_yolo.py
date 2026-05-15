"""
Conversione RDD2022 XML (Pascal VOC) -> YOLO per TUTTI i paesi.

Generalizzazione di convert_voc_to_yolo_japan.py: produce una directory per
ciascun client del Federated Learning, con il proprio data.yaml indipendente.

I 6 client seguono la convenzione del paper Saha et al. 2024 su RDD2022 + FL:
    Czech, India, Japan, Norway, United_States, China_MotorBike
(China_Drone e' escluso: dominio visivo molto diverso - foto da drone -
e non utilizzato nello standard di letteratura FL su RDD2022).

Output:
    data/processed/rdd2022_clients/
        czech/
            train/{images,labels}/
            valid/{images,labels}/
            data.yaml
            conversion_manifest.json
        india/
            ...
        japan/
            ...
        ...

Ogni client e' autonomo: puntandovi un comando Ultralytics si addestra solo
su quel paese. Questo e' esattamente cio' che ci serve nella Milestone 2 per
il client Flower.

Uso:
    python scripts/convert_voc_to_yolo_all.py
    python scripts/convert_voc_to_yolo_all.py --countries Japan India
    python scripts/convert_voc_to_yolo_all.py --val-fraction 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

# Le 4 classi standard di RDD2022 (Arya et al. 2022). L'indice YOLO segue
# l'ordine di questa lista. La classe critica esogena (Componente A) e' D40.
RDD_CLASSES: list[str] = ["D00", "D10", "D20", "D40"]
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(RDD_CLASSES)}

# I 6 client del Federated Learning. I nomi corrispondono ai folder del
# dataset RDD2022 ufficiale. China_Drone e' escluso (vedi docstring).
DEFAULT_COUNTRIES: list[str] = [
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
    "China_MotorBike",
]

# Path del progetto. Modifica RAW_ROOT se sposti il dataset altrove.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "rdd2022" / "RDD2022"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "rdd2022_clients"


# ---------------------------------------------------------------------------
# CONVERSIONE DI UNA SINGOLA ANNOTAZIONE
# ---------------------------------------------------------------------------

def convert_annotation(xml_path: Path) -> tuple[list[str], dict[str, int], str | None]:
    """Converte un singolo XML in righe YOLO.

    Returns:
        Tupla (yolo_lines, class_counts, error).
        - yolo_lines: righe del file .txt YOLO (lista vuota se nessuna bbox standard).
        - class_counts: conteggio di tutte le classi viste (anche quelle scartate).
        - error: stringa di errore se il file e' malformato, None altrimenti.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        return [], {}, f"parse_error: {e}"

    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return [], {}, "no_size_tag"

    try:
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
    except (AttributeError, TypeError, ValueError):
        return [], {}, "invalid_size_tag"

    if img_w == 0 or img_h == 0:
        return [], {}, "zero_size"

    yolo_lines: list[str] = []
    class_counts: dict[str, int] = {}

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or name_elem.text is None:
            continue
        class_name = name_elem.text.strip()
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Scarta le classi non standard (D01, D11, D43, D44, D50...)
        if class_name not in CLASS_TO_ID:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue

        # Conversione VOC -> YOLO normalizzata
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Sanity check
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1
                and 0 < width <= 1 and 0 < height <= 1):
            continue

        class_id = CLASS_TO_ID[class_name]
        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_lines, class_counts, None


# ---------------------------------------------------------------------------
# CONVERSIONE DI UN INTERO PAESE
# ---------------------------------------------------------------------------

def find_image_for_xml(xml_path: Path, img_dir: Path) -> Path | None:
    """Trova l'immagine corrispondente all'XML provando le estensioni comuni."""
    stem = xml_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_country(
    country: str,
    output_root: Path,
    val_fraction: float,
    seed: int,
) -> dict | None:
    """Converte un paese in un client FL pronto per Ultralytics.

    Returns:
        Dizionario con statistiche del paese, oppure None se la sorgente
        non esiste (paese saltato).
    """
    src_country_dir = RAW_ROOT / country
    src_xml_dir = src_country_dir / "train" / "annotations" / "xmls"
    src_img_dir = src_country_dir / "train" / "images"

    if not src_xml_dir.exists() or not src_img_dir.exists():
        print(f"[{country}] saltato: sorgente non trovata in {src_country_dir}")
        return None

    # Output con nome lowercase per coerenza file system (Linux/Windows)
    client_name = country.lower()
    client_root = output_root / client_name
    if client_root.exists():
        shutil.rmtree(client_root)

    xml_files = sorted(src_xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"[{country}] saltato: nessun XML trovato")
        return None

    print(f"[{country}] {len(xml_files)} XML trovati")

    # Split deterministico train/val (per-paese, seed costante)
    rng = random.Random(seed)
    indices = list(range(len(xml_files)))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_fraction)
    val_indices = set(indices[:n_val])

    for split in ("train", "valid"):
        (client_root / split / "images").mkdir(parents=True, exist_ok=True)
        (client_root / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "country": country,
        "client_name": client_name,
        "train": {"images": 0, "boxes": 0, "empty_labels": 0,
                   "boxes_per_class": {c: 0 for c in RDD_CLASSES}},
        "valid": {"images": 0, "boxes": 0, "empty_labels": 0,
                   "boxes_per_class": {c: 0 for c in RDD_CLASSES}},
        "errors": {"missing_image": 0, "parse_error": 0,
                   "no_size_tag": 0, "invalid_size_tag": 0, "zero_size": 0},
        "all_classes_seen": {},
    }

    for idx, xml_path in enumerate(xml_files):
        split = "valid" if idx in val_indices else "train"

        img_path = find_image_for_xml(xml_path, src_img_dir)
        if img_path is None:
            stats["errors"]["missing_image"] += 1
            continue

        yolo_lines, class_counts, error = convert_annotation(xml_path)

        for cls, count in class_counts.items():
            stats["all_classes_seen"][cls] = (
                stats["all_classes_seen"].get(cls, 0) + count
            )

        if error is not None:
            stats["errors"][error] = stats["errors"].get(error, 0) + 1
            continue

        # Copia immagine + scrivi label (anche vuota: YOLO accetta "negative samples")
        dst_img = client_root / split / "images" / img_path.name
        dst_lbl = client_root / split / "labels" / f"{xml_path.stem}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

        stats[split]["images"] += 1
        if yolo_lines:
            stats[split]["boxes"] += len(yolo_lines)
            # Conta bbox per classe nel split corrente
            for line in yolo_lines:
                cls_id = int(line.split()[0])
                cls_name = RDD_CLASSES[cls_id]
                stats[split]["boxes_per_class"][cls_name] += 1
        else:
            stats[split]["empty_labels"] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  [{country}] {idx + 1}/{len(xml_files)}")

    # data.yaml per Ultralytics
    yaml_content = (
        f"# Auto-generato da convert_voc_to_yolo_all.py\n"
        f"# Client FL: {client_name} (paese: {country})\n"
        f"path: {client_root.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"\n"
        f"nc: {len(RDD_CLASSES)}\n"
        f"names: {RDD_CLASSES}\n"
    )
    (client_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # Manifest dettagliato
    (client_root / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "classes_kept": RDD_CLASSES,
                "val_fraction": val_fraction,
                "seed": seed,
                "stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries", nargs="+", default=DEFAULT_COUNTRIES,
        help=f"Paesi da convertire. Default: {DEFAULT_COUNTRIES}",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Frazione del train da usare come validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed per lo split train/val deterministico (default: 42)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Salta la conferma di sovrascrittura",
    )
    return parser.parse_args()


def print_global_summary(all_stats: list[dict]) -> None:
    """Tabella riassuntiva multi-paese."""
    print("\n" + "=" * 80)
    print("RIEPILOGO GLOBALE")
    print("=" * 80)

    header = f"{'Country':<18} {'Train':>8} {'Valid':>8} {'D00':>7} {'D10':>7} {'D20':>7} {'D40*':>7}"
    print(header)
    print("-" * len(header))

    totals = {"train": 0, "valid": 0,
              "boxes_per_class": {c: 0 for c in RDD_CLASSES}}

    for s in all_stats:
        train_n = s["train"]["images"]
        valid_n = s["valid"]["images"]
        # Bbox totali per classe (train + valid) per il paese
        per_class = {
            c: s["train"]["boxes_per_class"][c] + s["valid"]["boxes_per_class"][c]
            for c in RDD_CLASSES
        }
        print(f"{s['country']:<18} {train_n:>8} {valid_n:>8} "
              f"{per_class['D00']:>7} {per_class['D10']:>7} "
              f"{per_class['D20']:>7} {per_class['D40']:>7}")

        totals["train"] += train_n
        totals["valid"] += valid_n
        for c in RDD_CLASSES:
            totals["boxes_per_class"][c] += per_class[c]

    print("-" * len(header))
    print(f"{'TOTAL':<18} {totals['train']:>8} {totals['valid']:>8} "
          f"{totals['boxes_per_class']['D00']:>7} "
          f"{totals['boxes_per_class']['D10']:>7} "
          f"{totals['boxes_per_class']['D20']:>7} "
          f"{totals['boxes_per_class']['D40']:>7}")
    print("\n* D40 (pothole) = classe critica esogena (Componente A del protocollo)")


def main() -> int:
    args = parse_args()

    print(f"Conversione RDD2022 -> YOLO format ({len(args.countries)} client)")
    print(f"  source : {RAW_ROOT}")
    print(f"  output : {OUTPUT_ROOT}")
    print(f"  classes: {RDD_CLASSES} (le altre vengono scartate)")
    print(f"  paesi  : {args.countries}")
    print(f"  val_fraction: {args.val_fraction}, seed: {args.seed}")
    print("-" * 80)

    if OUTPUT_ROOT.exists() and not args.force:
        ans = input(f"{OUTPUT_ROOT} esiste gia'. Sovrascrivere? [y/N] ").strip().lower()
        if ans != "y":
            print("Annullato.")
            return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    for country in args.countries:
        stats = process_country(country, OUTPUT_ROOT, args.val_fraction, args.seed)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("ERRORE: nessun paese convertito.")
        return 1

    # Manifest globale per riproducibilita'
    global_manifest_path = OUTPUT_ROOT / "global_manifest.json"
    global_manifest_path.write_text(
        json.dumps(
            {
                "classes_kept": RDD_CLASSES,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "countries": args.countries,
                "per_country_stats": all_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print_global_summary(all_stats)
    print(f"\nManifest globale: {global_manifest_path}")
    print(f"Output: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())