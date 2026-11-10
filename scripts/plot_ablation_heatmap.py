"""
Script per rigenerare la heatmap dell'ablation cap x time_decay.

Due modalita' d'uso:
    1. --from-results  : legge i results.json della fase ablation dello sweep
                         e calcola media su tre seed per ogni cella
    2. --hardcoded     : usa i valori gia' pubblicati nel report (Tab. 15)
                         e rigenera solo il plot (utile per sistemarne l'aspetto)

Uso:
    # Legge dai results.json della run di sweep piu' recente
    python plot_ablation_heatmap.py --from-results \\
        --sweep-dir experiments/report_pipeline \\
        --out figures/ablation_cap_decay_heatmap.png

    # Usa i numeri di Tabella 15 del report
    python plot_ablation_heatmap.py --hardcoded \\
        --out figures/ablation_cap_decay_heatmap.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Valori di Tabella 15 del report (media su tre seed, 10 round)
# ----------------------------------------------------------------------
# Righe = cap in {0.3, 0.5, 0.7}
# Colonne = time_decay in {0.85, 0.95, 1.00}
HARDCODED_MEANS = np.array([
    [0.0899, 0.1001, 0.0968],   # cap = 0.3
    [0.0938, 0.1030, 0.0967],   # cap = 0.5
    [0.0967, 0.0989, 0.0936],   # cap = 0.7
])

CAP_VALUES = [0.3, 0.5, 0.7]
DECAY_VALUES = [0.85, 0.95, 1.00]


# ----------------------------------------------------------------------
# Modalita' 1: pesca dai results.json
# ----------------------------------------------------------------------
def load_from_results(sweep_dir: Path) -> np.ndarray:
    """
    Legge tutti i results.json nella fase ablation dello sweep piu' recente
    e aggrega la metrica critical_ap50 al round 10 come media sui tre seed.

    Assume struttura:
        sweep_dir/
            YYYYMMDD_HHMMSS/
                ablation/
                    nlp_cap{c}_td{d}_seed{s}/
                        results.json
    """
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Cartella sweep non trovata: {sweep_dir}")

    # Trova la cartella piu' recente (lo sweep usa timestamp come nome)
    run_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not run_dirs:
        raise RuntimeError(f"Nessuna run trovata in {sweep_dir}")
    latest = run_dirs[-1]
    print(f"Leggo da: {latest}")

    ablation_root = latest / "ablation"
    if not ablation_root.exists():
        raise FileNotFoundError(f"Fase ablation non trovata: {ablation_root}")

    # Accumula per cella (cap, decay) le tre repliche sui seed
    accum: dict[tuple[float, float], list[float]] = {}
    for res_path in ablation_root.rglob("results.json"):
        run_name = res_path.parent.name
        # Formato atteso: nlp_cap{c}_td{d}_seed{s}
        try:
            parts = run_name.split("_")
            cap = float(parts[1].replace("cap", ""))
            decay = float(parts[2].replace("td", ""))
        except (IndexError, ValueError):
            print(f"  salto run con nome non riconosciuto: {run_name}")
            continue

        with open(res_path) as f:
            payload = json.load(f)

        # Formato Flower: metrics_distributed[metric] = List[[round, value]]
        try:
            series = payload["metrics_distributed"]["critical_ap50"]
            rnd, value = max(series, key=lambda entry: entry[0])
            value = float(value)
            if rnd != 10:
                print(f"  WARNING: ultimo round = {rnd} (atteso 10) in {res_path}")
        except (KeyError, IndexError, TypeError):
            print(f"  risultato vuoto o malformato, salto: {res_path}")
            continue

        accum.setdefault((cap, decay), []).append(value)

    # Costruisci la matrice media
    means = np.zeros((len(CAP_VALUES), len(DECAY_VALUES)))
    for i, cap in enumerate(CAP_VALUES):
        for j, decay in enumerate(DECAY_VALUES):
            seeds = accum.get((cap, decay), [])
            if not seeds:
                print(f"  WARNING: nessun risultato per cap={cap}, decay={decay}")
                means[i, j] = np.nan
            else:
                means[i, j] = float(np.mean(seeds))
                if len(seeds) < 3:
                    print(f"  WARNING: solo {len(seeds)} seed per cap={cap}, decay={decay}")

    return means


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_heatmap(
    means: np.ndarray,
    out_path: Path,
    cmap: str = "viridis",
    annotate_default: bool = True,
) -> None:
    """
    Produce la heatmap e la salva su disco. Il default (cap=0.5, decay=1.00)
    viene evidenziato con un bordo nero se annotate_default e' True.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    im = ax.imshow(means, cmap=cmap, aspect="auto")

    # Tick etichettati con i valori dei parametri
    ax.set_xticks(range(len(DECAY_VALUES)))
    ax.set_xticklabels([f"{d:.2f}" for d in DECAY_VALUES])
    ax.set_yticks(range(len(CAP_VALUES)))
    ax.set_yticklabels([f"{c:.1f}" for c in CAP_VALUES])

    ax.set_xlabel("time_decay", fontsize=11)
    ax.set_ylabel("cap", fontsize=11)
    ax.set_title("Ablation - critical_ap50 vs (cap, time_decay)", fontsize=12)

    # Annota ogni cella con il valore numerico
    for i in range(len(CAP_VALUES)):
        for j in range(len(DECAY_VALUES)):
            val = means[i, j]
            if np.isnan(val):
                continue
            # Scegli bianco o nero in base alla luminosita' dello sfondo
            vmin, vmax = np.nanmin(means), np.nanmax(means)
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            color = "white" if norm < 0.6 else "black"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                color=color, fontsize=12
            )

    # Evidenzia la cella di default (cap=0.5, decay=1.00)
    if annotate_default:
        default_i = CAP_VALUES.index(0.5)
        default_j = DECAY_VALUES.index(1.00)
        rect = plt.Rectangle(
            (default_j - 0.47, default_i - 0.47), 0.94, 0.94,
            fill=False, edgecolor="white", linewidth=3,
            zorder=5,
        )
        ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("critical_ap50 (media sui tre seed)", fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap salvata in: {out_path}")

    # Stampa anche la matrice per log/debug
    print("\nValori usati:")
    print(f"{'':>6}" + "".join(f"{d:>10.2f}" for d in DECAY_VALUES))
    for i, cap in enumerate(CAP_VALUES):
        row = f"{cap:>6.1f}" + "".join(f"{means[i, j]:>10.4f}" for j in range(len(DECAY_VALUES)))
        print(row)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-results", action="store_true",
        help="Legge i valori dai results.json della sweep",
    )
    parser.add_argument(
        "--hardcoded", action="store_true",
        help="Usa i valori di Tabella 15 del report",
    )
    parser.add_argument(
        "--sweep-dir", type=Path,
        default=Path("experiments/report_pipeline"),
        help="Cartella dello sweep (default: experiments/report_pipeline)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("figures/ablation_cap_decay_heatmap.png"),
        help="Path di output per il PNG",
    )
    parser.add_argument(
        "--cmap", default="viridis",
        help="Colormap matplotlib (default: viridis; "
             "alternative sensate: 'YlOrRd', 'Greens', 'RdYlGn', 'coolwarm')",
    )
    parser.add_argument(
        "--no-default-box", action="store_true",
        help="Non evidenziare la cella di default",
    )

    args = parser.parse_args()

    if args.from_results == args.hardcoded:
        parser.error("Scegliere esattamente una fra --from-results e --hardcoded")

    if args.from_results:
        means = load_from_results(args.sweep_dir)
    else:
        means = HARDCODED_MEANS

    plot_heatmap(
        means,
        out_path=args.out,
        cmap=args.cmap,
        annotate_default=not args.no_default_box,
    )


if __name__ == "__main__":
    main()