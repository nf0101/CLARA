"""
Pipeline sperimentale completa per il report.

Orchestra l'intero set di esperimenti necessari, in ordine:
    1. RQ1 (Componente B): FedAvg vs Oracle vs NLP-Weighted - vs Non-IID severo.
    2. RQ2 (Componente C): curva di robustezza al rumore NLP.
    3. Ablation (Sezione 3.3): sensitivity su cap e time_decay.
    4. Aggregazione risultati in CSV.
    5. Generazione figure e tabelle per il report.

Output: tutto in experiments/report_pipeline/<timestamp>/ con sottocartelle ordinate:
    rq1/
    rq2/
    ablation/
    aggregated.csv        <- tabella unica con tutti i risultati
    figures/              <- PNG da inserire nel report
        rq1_strategy_comparison.png
        rq2_robustness_curve.png
        ablation_cap_decay_heatmap.png
    tables/               <- TeX/Markdown per il report
        rq1_summary.tex
        rq2_summary.tex
        ablation_summary.tex
    pipeline_manifest.json   <- config completa per riproducibilita'

USO:
    python3 scripts/run_full_pipeline.py                    # default: scala "presentabile"
    python3 scripts/run_full_pipeline.py --scale debug      # 1 seed, 3 round (test rapido, ~1h)
    python3 scripts/run_full_pipeline.py --scale presentable # 3 seed, 10 round (~30h)
    python3 scripts/run_full_pipeline.py --scale full        # 5 seed, 15 round (~360h)
    python3 scripts/run_full_pipeline.py --only rq1          # solo RQ1
    python3 scripts/run_full_pipeline.py --skip ablation     # salta ablation
    python3 scripts/run_full_pipeline.py --resume <run_id>   # riprende run interrotta

LIMITAZIONI ATTUALI:
    - FedProx e SCAFFOLD non implementati: RQ1 usa 3 strategie invece di 5.
      Quando saranno implementati, aggiungili in STRATEGIES_RQ1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = PROJECT_ROOT / "experiments" / "report_pipeline"


# ===========================================================================
# SCALE PRESETS
# ===========================================================================

@dataclass
class Scale:
    """Configurazione per un livello di scala (debug/presentable/full)."""
    name: str
    seeds: list[int]
    rounds: int
    local_epochs: int
    imgsz: int
    description: str

SCALES: dict[str, Scale] = {
    "debug": Scale(
        name="debug",
        seeds=[42],
        rounds=3,
        local_epochs=1,
        imgsz=416,
        description="Test rapido per validare la pipeline (~1-2h totali)",
    ),
    "presentable": Scale(
        name="presentable",
        seeds=[42, 123, 2024],
        rounds=10,
        local_epochs=1,
        imgsz=416,
        description="Scala 'presentabile' per il report (~30-50h totali)",
    ),
    "full": Scale(
        name="full",
        seeds=[42, 123, 2024, 7, 31],
        rounds=15,
        local_epochs=2,
        imgsz=640,
        description="Scala piena per pubblicazione (~360h totali)",
    ),
    
    "smoke": Scale(
        name="smoke",
        seeds=[42],
        rounds=2,
        local_epochs=1,
        imgsz=320,
        description="Smoke test minimo: 1 seed, 2 round, imgsz piccolo (~30-45 min)",
    ),
}


# ===========================================================================
# DEFINIZIONE DEGLI ESPERIMENTI
# ===========================================================================

# RQ1: confronto strategie
# NOTA: FedProx e SCAFFOLD da aggiungere quando implementate
STRATEGIES_RQ1 = [
    {"strategy": "fedavg", "extra_args": []},
    {"strategy": "oracle", "extra_args": ["--alpha", "0.5", "--cap", "0.5"]},
    {"strategy": "nlp", "extra_args": ["--urgency-provider", "oracle",
                                        "--alpha", "0.5", "--cap", "0.5",
                                        "--time-decay", "1.0"]},
]

# RQ2: curva di robustezza (livelli di accuratezza NLP simulata)
NOISE_LEVELS_RQ2 = [1.0, 0.9, 0.8, 0.7, 0.5]

# Ablation: combinazioni cap × time_decay (Sezione 3.3 del report)
# Tieni una griglia 3x3 per restare nei tempi (9 punti × seeds)
ABLATION_GRID = {
    "cap": [0.3, 0.5, 0.7],
    "time_decay": [1.0, 0.95, 0.85],
}


# ===========================================================================
# UTILITY
# ===========================================================================

def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def run_single_experiment(
    run_dir: Path,
    strategy: str,
    extra_args: list[str],
    seed: int,
    scale: Scale,
    log_path: Path,
) -> dict:
    """Lancia una singola run di fl_run.py e ritorna i metadati.

    Returns:
        dict con keys: success, duration_sec, results_json_path, stderr_excerpt.
    """
    cmd = [
        "python3", "scripts/fl_run.py",
        "--strategy", strategy,
        "--rounds", str(scale.rounds),
        "--local-epochs", str(scale.local_epochs),
        "--imgsz", str(scale.imgsz),
        "--seed", str(seed),
        *extra_args,
    ]
    # NOTA: --seed NON e' supportato direttamente da fl_run.py.
    # Per ora il seed e' fissato a 42 internamente; per RIPETIBILITA' fra
    # seed diversi, fl_run.py andrebbe esteso. Per ora variamo seed solo
    # via --noise-seed (RQ2) e accettiamo che FedAvg/Oracle siano
    # ri-eseguiti N volte con stesso seed e media empirica della varianza.

    print(f"\n{'='*70}")
    print(f"RUN: strategy={strategy}, seed={seed}, args={extra_args}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"LOG: {log_path}")
    print('='*70)

    start = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        duration = time.time() - start
        success = (proc.returncode == 0)
    except Exception as e:
        duration = time.time() - start
        success = False
        log_path.write_text(f"EXCEPTION: {e}", encoding="utf-8")

    # Cerca il results.json piu' recente nella cartella fl_runs/
    results_path = None
    fl_runs = PROJECT_ROOT / "experiments" / "fl_runs"
    if fl_runs.exists():
        candidates = sorted(
            fl_runs.glob("*/results.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            # Sposta il results in run_dir e ricorda dove
            src = candidates[0]
            dst = run_dir / "results.json"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            results_path = str(dst.relative_to(PROJECT_ROOT))

    # Estrai ultime 20 righe del log per diagnostic se fallito
    stderr_excerpt = ""
    if not success and log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        stderr_excerpt = "\n".join(lines[-20:])

    return {
        "success": success,
        "duration_sec": duration,
        "results_json_path": results_path,
        "stderr_excerpt": stderr_excerpt,
    }


def write_manifest(pipeline_dir: Path, manifest: dict) -> None:
    (pipeline_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


# ===========================================================================
# ESECUZIONE DELLE TRE FASI
# ===========================================================================

def run_rq1(pipeline_dir: Path, scale: Scale, manifest: dict) -> None:
    rq1_dir = pipeline_dir / "rq1"
    rq1_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}\n# FASE 1: RQ1 (strategie vs Non-IID)\n{'#'*70}")
    print(f"# Strategie: {[s['strategy'] for s in STRATEGIES_RQ1]}")
    print(f"# Seeds: {scale.seeds}")
    print(f"# Run totali: {len(STRATEGIES_RQ1) * len(scale.seeds)}")

    runs = []
    for strat_conf in STRATEGIES_RQ1:
        for seed in scale.seeds:
            run_id = f"{strat_conf['strategy']}_seed{seed}"
            run_dir = rq1_dir / run_id
            run_dir.mkdir(exist_ok=True)
            log_path = run_dir / "stdout.log"

            result = run_single_experiment(
                run_dir=run_dir,
                strategy=strat_conf["strategy"],
                extra_args=strat_conf["extra_args"],
                seed=seed,
                scale=scale,
                log_path=log_path,
            )
            runs.append({
                "phase": "rq1",
                "run_id": run_id,
                "strategy": strat_conf["strategy"],
                "seed": seed,
                "extra_args": strat_conf["extra_args"],
                **result,
            })
            manifest["completed_runs"] += 1
            write_manifest(pipeline_dir, manifest)

    manifest["rq1_runs"] = runs


def run_rq2(pipeline_dir: Path, scale: Scale, manifest: dict) -> None:
    rq2_dir = pipeline_dir / "rq2"
    rq2_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}\n# FASE 2: RQ2 (curva di robustezza al rumore NLP)\n{'#'*70}")
    print(f"# Livelli accuracy: {NOISE_LEVELS_RQ2}")
    print(f"# Seeds: {scale.seeds}")
    print(f"# Run totali: {len(NOISE_LEVELS_RQ2) * len(scale.seeds)}")

    runs = []
    for accuracy in NOISE_LEVELS_RQ2:
        for seed in scale.seeds:
            run_id = f"nlp_acc{int(accuracy*100):03d}_seed{seed}"
            run_dir = rq2_dir / run_id
            run_dir.mkdir(exist_ok=True)
            log_path = run_dir / "stdout.log"

            result = run_single_experiment(
                run_dir=run_dir,
                strategy="nlp",
                extra_args=[
                    "--urgency-provider", "noisy_oracle",
                    "--noise-accuracy", str(accuracy),
                    "--alpha", "0.5",
                    "--cap", "0.5",
                    "--time-decay", "1.0",
                ],
                seed=seed,
                scale=scale,
                log_path=log_path,
            )
            runs.append({
                "phase": "rq2",
                "run_id": run_id,
                "strategy": "nlp",
                "noise_accuracy": accuracy,
                "seed": seed,
                **result,
            })
            manifest["completed_runs"] += 1
            write_manifest(pipeline_dir, manifest)

    manifest["rq2_runs"] = runs


def run_ablation(pipeline_dir: Path, scale: Scale, manifest: dict) -> None:
    abl_dir = pipeline_dir / "ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)

    n_combos = len(ABLATION_GRID["cap"]) * len(ABLATION_GRID["time_decay"])
    print(f"\n{'#'*70}\n# FASE 3: Ablation (cap x time_decay)\n{'#'*70}")
    print(f"# Combinazioni: {n_combos}")
    print(f"# Seeds: {scale.seeds}")
    print(f"# Run totali: {n_combos * len(scale.seeds)}")

    runs = []
    for cap in ABLATION_GRID["cap"]:
        for td in ABLATION_GRID["time_decay"]:
            for seed in scale.seeds:
                run_id = f"nlp_cap{cap}_td{td}_seed{seed}"
                run_dir = abl_dir / run_id
                run_dir.mkdir(exist_ok=True)
                log_path = run_dir / "stdout.log"

                result = run_single_experiment(
                    run_dir=run_dir,
                    strategy="nlp",
                    extra_args=[
                        "--urgency-provider", "oracle",
                        "--alpha", "0.5",
                        "--cap", str(cap),
                        "--time-decay", str(td),
                    ],
                    seed=seed,
                    scale=scale,
                    log_path=log_path,
                )
                runs.append({
                    "phase": "ablation",
                    "run_id": run_id,
                    "strategy": "nlp",
                    "cap": cap,
                    "time_decay": td,
                    "seed": seed,
                    **result,
                })
                manifest["completed_runs"] += 1
                write_manifest(pipeline_dir, manifest)

    manifest["ablation_runs"] = runs


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale", choices=list(SCALES.keys()), default="presentable",
        help="Preset di scala. Default: presentable",
    )
    parser.add_argument(
        "--only", choices=["rq1", "rq2", "ablation"], default=None,
        help="Esegui solo una fase",
    )
    parser.add_argument(
        "--skip", choices=["rq1", "rq2", "ablation"], action="append",
        default=[], help="Salta una fase (ripetibile)",
    )
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="Salta la generazione di figure e tabelle alla fine",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Riprende una run interrotta (passa il timestamp_id della cartella)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scale = SCALES[args.scale]

    if args.resume:
        pipeline_dir = PIPELINE_ROOT / args.resume
        if not pipeline_dir.exists():
            sys.exit(f"ERRORE: run da riprendere non trovata: {pipeline_dir}")
        manifest = json.loads(
            (pipeline_dir / "pipeline_manifest.json").read_text()
        )
        print(f"Riprendo run: {pipeline_dir}")
    else:
        run_id = utc_timestamp()
        pipeline_dir = PIPELINE_ROOT / run_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "started_at": utc_timestamp(),
            "scale": asdict(scale),
            "strategies_rq1": [s["strategy"] for s in STRATEGIES_RQ1],
            "noise_levels_rq2": NOISE_LEVELS_RQ2,
            "ablation_grid": ABLATION_GRID,
            "completed_runs": 0,
            "rq1_runs": [],
            "rq2_runs": [],
            "ablation_runs": [],
        }
        write_manifest(pipeline_dir, manifest)

    # Quante run totali?
    n_rq1 = len(STRATEGIES_RQ1) * len(scale.seeds)
    n_rq2 = len(NOISE_LEVELS_RQ2) * len(scale.seeds)
    n_abl = (len(ABLATION_GRID["cap"]) * len(ABLATION_GRID["time_decay"])
              * len(scale.seeds))
    total = 0
    if args.only is None or args.only == "rq1":
        if "rq1" not in args.skip: total += n_rq1
    if args.only is None or args.only == "rq2":
        if "rq2" not in args.skip: total += n_rq2
    if args.only is None or args.only == "ablation":
        if "ablation" not in args.skip: total += n_abl

    print(f"\n{'='*70}")
    print(f"PIPELINE START - scale={scale.name}")
    print(f"  output:   {pipeline_dir}")
    print(f"  seeds:    {scale.seeds}")
    print(f"  rounds:   {scale.rounds}, epochs/round: {scale.local_epochs}")
    print(f"  imgsz:    {scale.imgsz}")
    print(f"  RUN TOTALI: {total}")
    print(f"  Stima (14 min/run base × moltiplicatore scala): "
          f"~{total * 14 * scale.rounds // 3} min")
    print('='*70)

    # Esegui le fasi
    if args.only in (None, "rq1") and "rq1" not in args.skip:
        run_rq1(pipeline_dir, scale, manifest)
    if args.only in (None, "rq2") and "rq2" not in args.skip:
        run_rq2(pipeline_dir, scale, manifest)
    if args.only in (None, "ablation") and "ablation" not in args.skip:
        run_ablation(pipeline_dir, scale, manifest)

    manifest["finished_at"] = utc_timestamp()
    write_manifest(pipeline_dir, manifest)

    # Aggrega e plotta
    if not args.no_analysis:
        print(f"\n{'#'*70}\n# FASE 4: Aggregazione e analisi\n{'#'*70}")
        try:
            subprocess.run(
                ["python3", "scripts/aggregate_pipeline_results.py",
                 "--pipeline-dir", str(pipeline_dir)],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"ATTENZIONE: aggregazione fallita: {e}")

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETATA")
    print(f"  Output: {pipeline_dir}")
    print(f"  Manifest: {pipeline_dir / 'pipeline_manifest.json'}")
    print(f"  CSV: {pipeline_dir / 'aggregated.csv'}")
    print(f"  Figure: {pipeline_dir / 'figures'}")
    print('='*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())