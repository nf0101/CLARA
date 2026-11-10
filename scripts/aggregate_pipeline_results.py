"""
Aggregazione dei risultati della pipeline e generazione figure/tabelle.

Legge i results.json di tutte le run di una pipeline e produce:

    aggregated.csv           tabella unica long-format con tutti i risultati
    figures/
        rq1_strategy_comparison.png      bar plot critical_ap50 e mAP@50 per strategia (± std)
        rq1_per_round.png                curve critical_ap50 per round per strategia
        rq2_robustness_curve.png         critical_ap50 vs noise level (Oracle + FedAvg come riferimento)
        ablation_cap_decay_heatmap.png   heatmap critical_ap50 (cap x time_decay)
    tables/
        rq1_summary.tex                  tabella LaTeX per il report
        rq2_summary.tex
        ablation_summary.tex
        summary.md                       riassunto markdown leggibile

Uso:
    python3 scripts/aggregate_pipeline_results.py --pipeline-dir experiments/report_pipeline/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ===========================================================================
# CARICAMENTO E LONG-FORMAT
# ===========================================================================

def load_manifest(pipeline_dir: Path) -> dict:
    p = pipeline_dir / "pipeline_manifest.json"
    if not p.exists():
        sys.exit(f"ERRORE: manifest non trovato: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def build_long_dataframe(manifest: dict, pipeline_dir: Path) -> pd.DataFrame:
    """Costruisce un dataframe long-format con una riga per (run, round, metric).

    Colonne:
        phase, run_id, strategy, seed, noise_accuracy, cap, time_decay,
        round, metric, value, duration_sec
    """
    rows = []
    all_runs = (manifest.get("rq1_runs", []) +
                manifest.get("rq2_runs", []) +
                manifest.get("ablation_runs", []))

    for run in all_runs:
        if not run.get("success") or not run.get("results_json_path"):
            continue
        rjson_path = pipeline_dir.parent.parent / run["results_json_path"]
        # Adattamento path: results_json_path è relativo a PROJECT_ROOT
        # ma il manifest è dentro pipeline_dir, quindi normalizziamo:
        candidate = Path(run["results_json_path"])
        if not candidate.is_absolute():
            candidate = pipeline_dir.parents[1] / candidate
        if not candidate.exists():
            # fallback: cerca dentro la cartella della run
            run_dir_name = run["run_id"]
            for phase_dir in ("rq1", "rq2", "ablation"):
                fb = pipeline_dir / phase_dir / run_dir_name / "results.json"
                if fb.exists():
                    candidate = fb
                    break
        if not candidate.exists():
            print(f"  WARN: results.json non trovato per {run['run_id']}")
            continue

        data = json.loads(candidate.read_text(encoding="utf-8"))
        metrics = data.get("metrics_distributed", {})

        for metric_name, values in metrics.items():
            if metric_name == "partition_id":
                continue
            for round_idx, value in values:
                rows.append({
                    "phase": run["phase"],
                    "run_id": run["run_id"],
                    "strategy": run.get("strategy"),
                    "seed": run.get("seed"),
                    "noise_accuracy": run.get("noise_accuracy"),
                    "cap": run.get("cap"),
                    "time_decay": run.get("time_decay"),
                    "round": round_idx,
                    "metric": metric_name,
                    "value": value,
                    "duration_sec": run.get("duration_sec"),
                })

    return pd.DataFrame(rows)


# ===========================================================================
# AGGREGAZIONE STATISTICA
# ===========================================================================

def final_round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Restituisce solo le metriche all'ultimo round per ciascuna run."""
    if df.empty:
        return df
    last_round = df.groupby("run_id")["round"].transform("max")
    return df[df["round"] == last_round].copy()


def summarize_by_strategy(df_final: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggrega per strategia: mean, std, n sui seed."""
    sub = df_final[df_final["metric"] == metric]
    g = sub.groupby("strategy")["value"]
    return pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(),
        "n": g.count(),
    }).round(4)


def summarize_by_noise(df_final: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df_final[(df_final["metric"] == metric) &
                   (df_final["phase"] == "rq2")]
    g = sub.groupby("noise_accuracy")["value"]
    return pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(),
        "n": g.count(),
    }).round(4)


def summarize_ablation(df_final: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df_final[(df_final["metric"] == metric) &
                   (df_final["phase"] == "ablation")]
    g = sub.groupby(["cap", "time_decay"])["value"]
    return pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(),
        "n": g.count(),
    }).round(4)


# ===========================================================================
# FIGURE
# ===========================================================================

def plot_rq1_comparison(df_final: pd.DataFrame, figures_dir: Path) -> None:
    """Bar plot critical_ap50 e mAP@50 per strategia con barre di errore."""
    rq1 = df_final[df_final["phase"] == "rq1"]
    if rq1.empty:
        return

    metrics_to_plot = ["critical_ap50", "map50", "map50_95"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        sub = rq1[rq1["metric"] == metric]
        g = sub.groupby("strategy")["value"].agg(["mean", "std"])
        if g.empty:
            ax.set_title(f"{metric}: no data"); continue
        x = np.arange(len(g))
        ax.bar(x, g["mean"], yerr=g["std"].fillna(0),
               capsize=8, color=["#888888", "#2ca02c", "#1f77b4"][:len(g)])
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=15)
        ax.set_title(metric)
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("RQ1 - Confronto strategie (media ± std sui seed)", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "rq1_strategy_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_rq1_per_round(df: pd.DataFrame, figures_dir: Path) -> None:
    """Curve critical_ap50 vs round, una linea per strategia, media sui seed."""
    rq1 = df[(df["phase"] == "rq1") & (df["metric"] == "critical_ap50")]
    if rq1.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, sub in rq1.groupby("strategy"):
        g = sub.groupby("round")["value"].agg(["mean", "std"]).reset_index()
        ax.plot(g["round"], g["mean"], marker="o", label=strategy, linewidth=2)
        ax.fill_between(g["round"],
                         g["mean"] - g["std"].fillna(0),
                         g["mean"] + g["std"].fillna(0),
                         alpha=0.2)
    ax.set_xlabel("Round federato")
    ax.set_ylabel("critical_ap50 (D40)")
    ax.set_title("RQ1 - critical_ap50 per round (media ± std sui seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "rq1_per_round.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rq2_curve(df_final: pd.DataFrame, figures_dir: Path) -> None:
    """Curva critical_ap50 vs noise accuracy + linea FedAvg e Oracle come riferimento."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # RQ2: punti per livello di accuracy
    rq2 = df_final[(df_final["phase"] == "rq2") &
                   (df_final["metric"] == "critical_ap50")]
    if not rq2.empty:
        g = rq2.groupby("noise_accuracy")["value"].agg(["mean", "std"]).reset_index()
        g = g.sort_values("noise_accuracy", ascending=False)
        ax.errorbar(g["noise_accuracy"], g["mean"], yerr=g["std"].fillna(0),
                    marker="o", capsize=5, linewidth=2, label="NLP-Weighted (con rumore)")

    # Linee di riferimento da RQ1
    rq1 = df_final[(df_final["phase"] == "rq1") &
                   (df_final["metric"] == "critical_ap50")]
    for ref_strategy, color in [("fedavg", "gray"), ("oracle", "green")]:
        sub = rq1[rq1["strategy"] == ref_strategy]
        if sub.empty: continue
        m = sub["value"].mean()
        s = sub["value"].std() if len(sub) > 1 else 0
        ax.axhline(m, color=color, linestyle="--",
                   label=f"{ref_strategy} (mean={m:.3f})")
        ax.fill_between([0.4, 1.05], m-s, m+s, color=color, alpha=0.1)

    ax.set_xlabel("Accuratezza simulata del classificatore NLP")
    ax.set_ylabel("critical_ap50 (D40)")
    ax.set_title("RQ2 - Curva di robustezza al rumore NLP")
    ax.set_xlim(0.4, 1.05)
    ax.invert_xaxis()    # da sinistra (accuracy alta) verso destra (bassa)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "rq2_robustness_curve.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmap(df_final: pd.DataFrame, figures_dir: Path) -> None:
    """Heatmap critical_ap50 sulla griglia cap × time_decay."""
    abl = df_final[(df_final["phase"] == "ablation") &
                   (df_final["metric"] == "critical_ap50")]
    if abl.empty:
        return

    pivot = abl.groupby(["cap", "time_decay"])["value"].mean().unstack("time_decay")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
    ax.set_xlabel("time_decay")
    ax.set_ylabel("cap")
    ax.set_title("Ablation - critical_ap50 vs (cap, time_decay)")

    # annotazioni
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                    ha="center", va="center",
                    color="white" if pivot.values[i,j] < pivot.values.mean() else "black")

    # evidenzia la configurazione di default usata in RQ1 e RQ2
    DEFAULT_CAP, DEFAULT_DECAY = 0.5, 1.00
    rows = [float(v) for v in pivot.index]
    cols = [float(v) for v in pivot.columns]
    i_def = min(range(len(rows)), key=lambda k: abs(rows[k] - DEFAULT_CAP))
    j_def = min(range(len(cols)), key=lambda k: abs(cols[k] - DEFAULT_DECAY))
    ax.add_patch(Rectangle((j_def - 0.47, i_def - 0.47), 0.94, 0.94,
                           fill=False, edgecolor="white",
                           linewidth=3, zorder=5))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_cap_decay_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# TABELLE (LaTeX + Markdown)
# ===========================================================================

def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    """Salva un dataframe come tabella LaTeX pronta per il report."""
    with path.open("w", encoding="utf-8") as f:
        f.write(f"% {caption}\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(df.to_latex(float_format="%.4f", escape=False))
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{tab:{label}}}\n")
        f.write("\\end{table}\n")


def write_markdown_summary(
    rq1_sum: pd.DataFrame,
    rq2_sum: pd.DataFrame,
    abl_sum: pd.DataFrame,
    manifest: dict,
    path: Path,
) -> None:
    lines = [
        "# Pipeline Results Summary",
        "",
        f"- Run ID: `{manifest.get('run_id', 'n/a')}`",
        f"- Scale: `{manifest.get('scale', {}).get('name', 'n/a')}`",
        f"- Started: {manifest.get('started_at', 'n/a')}",
        f"- Finished: {manifest.get('finished_at', 'n/a')}",
        f"- Total runs: {manifest.get('completed_runs', 0)}",
        "",
        "## RQ1: confronto strategie (critical_ap50 al round finale)",
        "",
        rq1_sum.to_markdown() if not rq1_sum.empty else "_no data_",
        "",
        "## RQ2: curva di robustezza al rumore NLP",
        "",
        rq2_sum.to_markdown() if not rq2_sum.empty else "_no data_",
        "",
        "## Ablation: cap × time_decay",
        "",
        abl_sum.to_markdown() if not abl_sum.empty else "_no data_",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-dir", type=Path, required=True,
                   help="Cartella di una run pipeline (experiments/report_pipeline/<id>)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pipeline_dir.exists():
        sys.exit(f"ERRORE: cartella non trovata: {args.pipeline_dir}")

    manifest = load_manifest(args.pipeline_dir)
    df = build_long_dataframe(manifest, args.pipeline_dir)
    if df.empty:
        sys.exit("ERRORE: nessun dato aggregato (forse tutte le run hanno fallito?)")

    # Salva il CSV long-format
    csv_path = args.pipeline_dir / "aggregated.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Long-format CSV: {csv_path} ({len(df)} righe)")

    # Aggregazioni
    df_final = final_round_metrics(df)
    rq1_summary = summarize_by_strategy(df_final, "critical_ap50")
    rq2_summary = summarize_by_noise(df_final, "critical_ap50")
    abl_summary = summarize_ablation(df_final, "critical_ap50")

    # Cartelle output
    figures_dir = args.pipeline_dir / "figures"
    tables_dir = args.pipeline_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    # Figure
    print("  Genero figure...")
    plot_rq1_comparison(df_final, figures_dir)
    plot_rq1_per_round(df, figures_dir)
    plot_rq2_curve(df_final, figures_dir)
    plot_ablation_heatmap(df_final, figures_dir)

    # Tabelle
    print("  Genero tabelle...")
    if not rq1_summary.empty:
        write_latex_table(rq1_summary, tables_dir / "rq1_summary.tex",
                           "RQ1: confronto strategie (critical_ap50 al round finale)",
                           "rq1_summary")
    if not rq2_summary.empty:
        write_latex_table(rq2_summary, tables_dir / "rq2_summary.tex",
                           "RQ2: robustezza al rumore NLP", "rq2_summary")
    if not abl_summary.empty:
        write_latex_table(abl_summary, tables_dir / "ablation_summary.tex",
                           "Ablation: cap x time_decay", "ablation")

    write_markdown_summary(rq1_summary, rq2_summary, abl_summary, manifest,
                            tables_dir / "summary.md")

    print(f"\n=========================================================")
    print(f" Aggregazione completata.")
    print(f"   CSV:    {csv_path}")
    print(f"   Figure: {figures_dir}")
    print(f"   Tabelle:{tables_dir}")
    print(f"   Summary: {tables_dir / 'summary.md'}")
    print(f"=========================================================")

    # Stampa il summary in output
    print("\n--- SUMMARY ---")
    if not rq1_summary.empty:
        print("\nRQ1:"); print(rq1_summary)
    if not rq2_summary.empty:
        print("\nRQ2:"); print(rq2_summary)
    if not abl_summary.empty:
        print("\nAblation:"); print(abl_summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())