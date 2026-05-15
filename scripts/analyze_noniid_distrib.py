"""
Analisi della distribuzione Non-IID classe x paese in RDD2022.

Legge il global_manifest.json prodotto da convert_voc_to_yolo_all.py e genera:

1. CSV con tre tabelle:
    a) conteggi assoluti di bbox (paese x classe)
    b) row-normalized (% delle bbox di un paese per ciascuna classe)
       -> "che mix di degrado vede ciascun client?"
    c) col-normalized (% delle bbox di una classe per ciascun paese)
       -> "in che paese si concentra ciascuna classe?"

2. Figura PNG con tre heatmap affiancate (assoluti, row-norm, col-norm).
   Documenta visivamente quanto sia severo il Non-IID, da inserire nel report.

3. Metriche di concentrazione per ciascuna classe (Gini, max share del client
   dominante) -> utili per discutere quanto sia "skewed" il dataset.

Uso:
    python scripts/analyze_noniid_distribution.py
    python scripts/analyze_noniid_distribution.py \\
        --manifest data/processed/rdd2022_clients/global_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "rdd2022_clients" / "global_manifest.json"
)
DEFAULT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CRITICAL_CLASS = "D40"   # pothole (Componente A)


# ---------------------------------------------------------------------------
# CARICAMENTO E COSTRUZIONE DELLA MATRICE
# ---------------------------------------------------------------------------

def build_distribution_matrix(manifest: dict) -> pd.DataFrame:
    """Costruisce la matrice paese x classe (conteggi assoluti, train + valid)."""
    classes = manifest["classes_kept"]
    rows = []
    index = []

    for stats in manifest["per_country_stats"]:
        country = stats["country"]
        index.append(country)
        per_class = {
            c: stats["train"]["boxes_per_class"][c]
               + stats["valid"]["boxes_per_class"][c]
            for c in classes
        }
        rows.append(per_class)

    return pd.DataFrame(rows, index=index, columns=classes)


# ---------------------------------------------------------------------------
# METRICHE DI CONCENTRAZIONE
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """Indice di Gini: 0 = uniforme, ~1 = concentrato in un solo client.

    Formula stabile basata sulla differenza media assoluta.
    """
    values = np.asarray(values, dtype=float)
    if values.sum() == 0:
        return float("nan")
    values_sorted = np.sort(values)
    n = len(values_sorted)
    cumulative = np.cumsum(values_sorted)
    # Formula standard
    return (2.0 * np.sum((np.arange(1, n + 1)) * values_sorted)
            / (n * cumulative[-1])) - (n + 1) / n


def compute_concentration_metrics(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per ogni classe, calcola:
       - share del client dominante,
       - secondo client (per misurare se la concentrazione e' su un solo
         o su pochi client),
       - Gini coefficient.
    """
    metrics = []
    for cls in matrix.columns:
        col = matrix[cls].values
        total = col.sum()
        if total == 0:
            metrics.append({"class": cls, "total": 0,
                             "top1_share": float("nan"),
                             "top1_country": "n/a",
                             "top2_share": float("nan"),
                             "gini": float("nan")})
            continue

        sorted_idx = np.argsort(col)[::-1]
        top1_share = col[sorted_idx[0]] / total
        top1_country = matrix.index[sorted_idx[0]]
        top2_share = col[sorted_idx[1]] / total if len(col) > 1 else 0.0

        metrics.append({
            "class": cls,
            "total": int(total),
            "top1_country": top1_country,
            "top1_share": round(float(top1_share), 4),
            "top2_share": round(float(top2_share), 4),
            "gini": round(float(gini_coefficient(col)), 4),
        })

    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------------
# OUTPUT: CSV
# ---------------------------------------------------------------------------

def save_csv_tables(
    abs_matrix: pd.DataFrame,
    row_norm: pd.DataFrame,
    col_norm: pd.DataFrame,
    concentration: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Salva le quattro tabelle in un singolo CSV con sezioni separate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "class_distribution.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Distribuzione bbox per classe e paese (RDD2022)\n")
        f.write(f"# Classe critica esogena: {CRITICAL_CLASS}\n\n")

        f.write("## Tabella 1: conteggi assoluti (bbox)\n")
        abs_matrix.to_csv(f)

        f.write("\n## Tabella 2: row-normalized (mix di classi per ciascun client)\n")
        f.write("# Ogni riga somma a 1.0\n")
        row_norm.round(4).to_csv(f)

        f.write("\n## Tabella 3: col-normalized (concentrazione di ciascuna classe sui client)\n")
        f.write("# Ogni colonna somma a 1.0\n")
        col_norm.round(4).to_csv(f)

        f.write("\n## Tabella 4: metriche di concentrazione per classe\n")
        f.write("# top1_share: frazione delle bbox della classe nel client dominante\n")
        f.write("# top2_share: frazione nel secondo client\n")
        f.write("# gini: 0 = uniforme su tutti i client, ~1 = concentrato in uno solo\n")
        concentration.to_csv(f, index=False)

    return csv_path


# ---------------------------------------------------------------------------
# OUTPUT: HEATMAP
# ---------------------------------------------------------------------------

def save_heatmaps(
    abs_matrix: pd.DataFrame,
    row_norm: pd.DataFrame,
    col_norm: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Tre heatmap affiancate per visualizzare il Non-IID."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "class_distribution_heatmap.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Helper per disegnare una heatmap con annotazioni
    def draw(ax, data, title, fmt, cmap):
        im = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        ax.set_title(title)
        ax.set_xlabel("Classe")

        # Annotazioni nelle celle
        vmax = data.values.max()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                color = "white" if val > vmax * 0.5 else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color=color, fontsize=9)

        # Evidenzia la colonna della classe critica
        if CRITICAL_CLASS in data.columns:
            j = list(data.columns).index(CRITICAL_CLASS)
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, data.shape[0],
                                        fill=False, edgecolor="red", lw=2))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    draw(axes[0], abs_matrix, "Conteggi assoluti (bbox)", "d", "viridis")
    draw(axes[1], row_norm, "Mix di classi per client (row-norm)", ".2f", "Blues")
    draw(axes[2], col_norm, "Concentrazione classi sui client (col-norm)",
          ".2f", "Reds")

    axes[0].set_ylabel("Client (paese)")
    fig.suptitle(
        f"Distribuzione Non-IID di RDD2022 (rosso = classe critica esogena: {CRITICAL_CLASS})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Path al global_manifest.json",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
        help="Directory dove salvare il CSV",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help="Directory dove salvare la heatmap",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.manifest.exists():
        sys.exit(f"ERRORE: manifest non trovato in {args.manifest}.\n"
                 f"Hai eseguito convert_voc_to_yolo_all.py?")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    abs_matrix = build_distribution_matrix(manifest)

    # Normalizzazioni
    row_norm = abs_matrix.div(abs_matrix.sum(axis=1), axis=0)
    col_norm = abs_matrix.div(abs_matrix.sum(axis=0), axis=1)

    concentration = compute_concentration_metrics(abs_matrix)

    csv_path = save_csv_tables(abs_matrix, row_norm, col_norm,
                                 concentration, args.tables_dir)
    fig_path = save_heatmaps(abs_matrix, row_norm, col_norm, args.figures_dir)

    # Stampa a video sintesi
    print("Distribuzione Non-IID di RDD2022")
    print("=" * 70)
    print("\nConteggi assoluti (bbox):")
    print(abs_matrix.to_string())

    print("\n\nRow-normalized (mix di classi per ciascun client):")
    print(row_norm.round(3).to_string())

    print("\n\nMetriche di concentrazione per classe:")
    print(concentration.to_string(index=False))

    print(f"\nCSV       : {csv_path}")
    print(f"Heatmap   : {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())